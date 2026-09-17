"""Tests for grok_fleet.queue — durable SQLite job queue.

Pure stdlib + pytest. Covers: enqueue idempotency, atomic claim, complete/park
state transitions, watchdog requeue of abandoned leases, and the crucial
concurrency guarantee that two racing workers never claim the same job.

Concurrency is exercised with real threads against separate connections to the
same on-disk DB (each JobQueue opens its own sqlite connection), which is how a
multi-worker deployment actually races. WAL + BEGIN IMMEDIATE must serialize the
claims.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import Counter

import pytest

from grok_fleet.queue import (
    STATUS_CLAIMED,
    STATUS_DONE,
    STATUS_PARKED,
    STATUS_QUEUED,
    JobQueue,
    new_worker_id,
)


@pytest.fixture()
def db_path():
    d = tempfile.mkdtemp(prefix="gqf_queue_")
    path = os.path.join(d, "queue.db")
    yield path
    # best-effort cleanup; WAL sidecar files may linger on Windows
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


@pytest.fixture()
def q(db_path):
    queue = JobQueue(db_path, lease_seconds=300)
    yield queue
    queue.close()


# --------------------------------------------------------------------------- #
# enqueue / claim basics
# --------------------------------------------------------------------------- #


def test_enqueue_then_claim_returns_job(q):
    q.enqueue("t1", '{"work": 1}')
    claimed = q.claim("worker-a")
    assert claimed == ("t1", '{"work": 1}')
    assert q.status_of("t1") == STATUS_CLAIMED


def test_claim_empty_queue_returns_none(q):
    assert q.claim("worker-a") is None


def test_enqueue_is_idempotent_on_task_id(q):
    q.enqueue("dup", "first")
    q.enqueue("dup", "second")  # ignored — original preserved
    claimed = q.claim("w")
    assert claimed == ("dup", "first")
    # only one job total
    assert q.counts() == {STATUS_CLAIMED: 1}


def test_enqueue_rejects_empty_task_id(q):
    with pytest.raises(ValueError):
        q.enqueue("", "payload")


def test_claim_rejects_empty_worker_id(q):
    with pytest.raises(ValueError):
        q.claim("")


def test_claim_is_fifo_by_created_at(q):
    q.enqueue("a", "pa")
    time.sleep(0.002)
    q.enqueue("b", "pb")
    first = q.claim("w")
    second = q.claim("w")
    assert first[0] == "a"
    assert second[0] == "b"


# --------------------------------------------------------------------------- #
# complete / park transitions
# --------------------------------------------------------------------------- #


def test_complete_marks_done(q):
    q.enqueue("t1", "p")
    q.claim("w")
    q.complete("t1", "result-text")
    assert q.status_of("t1") == STATUS_DONE


def test_complete_on_unclaimed_is_noop(q):
    q.enqueue("t1", "p")  # still queued, never claimed
    q.complete("t1", "r")  # no-op (only 'claimed' -> done)
    assert q.status_of("t1") == STATUS_QUEUED


def test_park_from_claimed(q):
    q.enqueue("t1", "p")
    q.claim("w")
    q.park("t1", "quality bar not met after 2 revises")
    assert q.status_of("t1") == STATUS_PARKED


def test_park_from_queued_allowed(q):
    q.enqueue("t1", "p")  # never claimed
    q.park("t1", "human shelved it")
    assert q.status_of("t1") == STATUS_PARKED


def test_park_on_done_is_noop(q):
    q.enqueue("t1", "p")
    q.claim("w")
    q.complete("t1", "r")
    q.park("t1", "too late")  # done is terminal for park
    assert q.status_of("t1") == STATUS_DONE


def test_claimed_job_not_reclaimable(q):
    q.enqueue("t1", "p")
    q.claim("w1")
    # a second worker finds nothing to claim
    assert q.claim("w2") is None


# --------------------------------------------------------------------------- #
# watchdog requeue of abandoned leases
# --------------------------------------------------------------------------- #


def test_watchdog_requeues_expired_lease(db_path):
    # short lease so it expires quickly
    q = JobQueue(db_path, lease_seconds=1)
    try:
        q.enqueue("t1", "p")
        q.claim("dead-worker")
        assert q.status_of("t1") == STATUS_CLAIMED
        # before expiry: watchdog does nothing
        assert q.watchdog_requeue() == 0
        # let the lease expire
        time.sleep(1.1)
        n = q.watchdog_requeue()
        assert n == 1
        assert q.status_of("t1") == STATUS_QUEUED
        # and it can be re-claimed by a live worker
        reclaimed = q.claim("live-worker")
        assert reclaimed == ("t1", "p")
    finally:
        q.close()


def test_watchdog_ignores_done_and_active(db_path):
    q = JobQueue(db_path, lease_seconds=1)
    try:
        # enqueue with a real time gap so FIFO order is unambiguous, and claim
        # the FIRST job so we know exactly which one we complete.
        q.enqueue("job_first", "p1")
        time.sleep(0.01)
        q.enqueue("job_second", "p2")
        claimed = q.claim("w")  # fifo -> job_first
        assert claimed[0] == "job_first"
        q.complete("job_first", "r")
        # job_second still queued, unclaimed -> no lease -> watchdog ignores it;
        # job_first is done -> also ignored. Nothing to requeue.
        time.sleep(1.1)
        assert q.watchdog_requeue() == 0
        assert q.status_of("job_first") == STATUS_DONE
        assert q.status_of("job_second") == STATUS_QUEUED
    finally:
        q.close()


def test_watchdog_after_close_returns_zero(db_path):
    q = JobQueue(db_path, lease_seconds=1)
    q.enqueue("t1", "p")
    q.claim("w")
    q.close()
    # operating on a closed connection must not raise; returns safe 0
    assert q.watchdog_requeue() == 0


# --------------------------------------------------------------------------- #
# THE concurrency guarantee: two workers never claim the same job
# --------------------------------------------------------------------------- #


def test_two_concurrent_claims_never_grab_same_job(db_path):
    # Arrange — one job, two workers racing for it.
    seed = JobQueue(db_path, lease_seconds=300)
    seed.enqueue("solo", "payload")
    seed.close()

    results = []
    barrier = threading.Barrier(2)

    def worker(worker_id: str):
        # each worker uses its OWN connection to the same db file
        wq = JobQueue(db_path, lease_seconds=300)
        try:
            barrier.wait()  # line them up to maximize the race
            got = wq.claim(worker_id)
            results.append(got)
        finally:
            wq.close()

    threads = [
        threading.Thread(target=worker, args=(new_worker_id(),)) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Assert — exactly one got the job, the other got None.
    got_jobs = [r for r in results if r is not None]
    got_none = [r for r in results if r is None]
    assert len(got_jobs) == 1, f"expected exactly one claimer, got {results}"
    assert len(got_none) == 1
    assert got_jobs[0] == ("solo", "payload")


def test_many_workers_partition_jobs_without_overlap(db_path):
    # Arrange — N jobs, M workers all draining concurrently.
    n_jobs = 60
    n_workers = 8
    seed = JobQueue(db_path, lease_seconds=300)
    for i in range(n_jobs):
        seed.enqueue(f"job-{i:03d}", f"payload-{i}")
    seed.close()

    claimed_ids = []
    lock = threading.Lock()
    start = threading.Barrier(n_workers)

    def drain(worker_id: str):
        wq = JobQueue(db_path, lease_seconds=300)
        try:
            start.wait()
            while True:
                got = wq.claim(worker_id)
                if got is None:
                    break
                with lock:
                    claimed_ids.append(got[0])
        finally:
            wq.close()

    threads = [
        threading.Thread(target=drain, args=(new_worker_id(),))
        for _ in range(n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Assert — every job claimed exactly once, no duplicates, none lost.
    counts = Counter(claimed_ids)
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"jobs claimed more than once: {dupes}"
    assert len(claimed_ids) == n_jobs
    assert set(claimed_ids) == {f"job-{i:03d}" for i in range(n_jobs)}


# --------------------------------------------------------------------------- #
# durability: reopen the db and state survives
# --------------------------------------------------------------------------- #


def test_state_survives_reopen(db_path):
    q1 = JobQueue(db_path, lease_seconds=300)
    q1.enqueue("persist", "p")
    q1.claim("w")
    q1.complete("persist", "done")
    q1.close()

    q2 = JobQueue(db_path, lease_seconds=300)
    try:
        assert q2.status_of("persist") == STATUS_DONE
    finally:
        q2.close()


def test_context_manager_closes(db_path):
    with JobQueue(db_path, lease_seconds=300) as q:
        q.enqueue("t1", "p")
        assert q.status_of("t1") == STATUS_QUEUED
    # after context exit, connection is closed; a fresh queue still reads it
    with JobQueue(db_path, lease_seconds=300) as q2:
        assert q2.status_of("t1") == STATUS_QUEUED


def test_bad_lease_seconds_rejected(db_path):
    with pytest.raises(ValueError):
        JobQueue(db_path, lease_seconds=0)
