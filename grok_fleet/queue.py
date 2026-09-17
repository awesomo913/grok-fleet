"""grok_fleet.queue — durable SQLite-backed job queue (stdlib sqlite3 only).

Implements the FROZEN ``JobQueue`` signature from ``grok_fleet.interfaces``.
Crash-safe: jobs move ``queued -> claimed -> (done | parked)`` and a claim
stamps a lease. A worker that dies mid-job leaves its lease to expire; the next
:meth:`JobQueue.watchdog_requeue` returns it to ``queued`` so no task is
silently lost.

Concurrency
-----------
Two guarantees the tests pin down:
  1. ``claim`` is ATOMIC — two workers racing on the same queue never take the
     same job. We use a single UPDATE ... RETURNING guarded by
     ``BEGIN IMMEDIATE`` (grabs the write lock up front) so exactly one writer
     mutates a row at a time. On sqlite older than 3.35 (no RETURNING) we fall
     back to a claim-token UPDATE then SELECT within the same immediate
     transaction, which is equally atomic.
  2. WAL journal mode + ``busy_timeout`` so concurrent connections queue on the
     write lock instead of erroring with "database is locked".

Robustness
----------
Every sqlite call is wrapped: we catch ``sqlite3.Error`` (and ``OSError`` for
disk-level failures), log a warning, and return a safe value (None / 0 / no-op)
rather than letting a DB error crash a worker or an HTTP handler. Per the
workspace rule: no bare excepts, always bind + log.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from typing import Optional, Tuple

log = logging.getLogger(__name__)

__all__ = ["JobQueue"]

# Job status values (named, not magic strings scattered around).
STATUS_QUEUED = "queued"
STATUS_CLAIMED = "claimed"
STATUS_DONE = "done"
STATUS_PARKED = "parked"

DEFAULT_LEASE_SECONDS = 300
_BUSY_TIMEOUT_MS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    task_id     TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    worker_id   TEXT,
    result      TEXT,
    reason      TEXT,
    lease_until REAL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created
    ON jobs (status, created_at);
"""


def _sqlite_supports_returning() -> bool:
    """True when the linked sqlite supports UPDATE ... RETURNING (>= 3.35)."""
    parts = sqlite3.sqlite_version_info
    return parts >= (3, 35, 0)


class JobQueue:
    """A crash-safe SQLite-backed job queue with claim/complete/park + watchdog.

    Jobs move through states: queued -> claimed -> (done | parked). A claim
    stamps a lease/heartbeat; the watchdog requeues jobs whose lease expired
    (worker died mid-run) so no task is silently lost.
    """

    def __init__(self, db_path: str, *, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> None:
        """Open/create the SQLite queue at ``db_path`` and ensure the schema.

        lease_seconds — how long a claim is valid before the watchdog may
        requeue it. Uses stdlib sqlite3 only; no ORM. Enables WAL journal mode
        for concurrent readers/writers and a busy timeout so racing writers
        block instead of erroring.
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")
        self._db_path = db_path
        self._lease_seconds = int(lease_seconds)
        self._use_returning = _sqlite_supports_returning()
        # Serialize this instance's own calls; cross-process/other-connection
        # atomicity is handled by BEGIN IMMEDIATE + WAL, not this lock.
        self._lock = threading.RLock()
        self._conn = self._connect(db_path)
        self._ensure_schema()

    # -- connection / schema -------------------------------------------------

    def _connect(self, db_path: str) -> sqlite3.Connection:
        """Open a connection with WAL + busy timeout. Isolation left manual."""
        # isolation_level=None => autocommit; we drive transactions explicitly
        # with BEGIN IMMEDIATE for the atomic claim path.
        conn = sqlite3.connect(
            db_path,
            timeout=_BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=%d;" % _BUSY_TIMEOUT_MS)
        except sqlite3.Error as exc:
            log.warning("PRAGMA setup failed for %s: %s", db_path, exc)
        return conn

    def _ensure_schema(self) -> None:
        """Create the jobs table + index if absent."""
        with self._lock:
            try:
                self._conn.executescript(_SCHEMA)
            except (sqlite3.Error, OSError) as exc:
                log.warning("schema init failed for %s: %s", self._db_path, exc)

    # -- write path ----------------------------------------------------------

    def enqueue(self, task_id: str, payload: str) -> None:
        """Insert a new queued job. ``payload`` is opaque JSON-ish text.

        Idempotent on task_id: re-enqueuing an existing id is a no-op (the
        original job is preserved), never a duplicate active job. We use
        INSERT OR IGNORE so a re-enqueue does not disturb an in-flight or
        completed job of the same id.
        """
        if not task_id:
            raise ValueError("task_id must be non-empty")
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO jobs "
                    "(task_id, payload, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (task_id, payload, STATUS_QUEUED, now, now),
                )
            except (sqlite3.Error, OSError) as exc:
                log.warning("enqueue failed for task_id=%r: %s", task_id, exc)

    def claim(self, worker_id: str) -> Optional[Tuple[str, str]]:
        """Atomically claim the oldest queued job for ``worker_id``.

        Returns (task_id, payload) and stamps a lease, or None if the queue is
        empty. The claim is atomic so two workers never take the same job: we
        open a BEGIN IMMEDIATE transaction (acquires the write lock before any
        read) and flip exactly one row from 'queued' to 'claimed'. Any sqlite
        error rolls back and returns None (the job stays queued for a retry).
        """
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        now = time.time()
        lease_until = now + self._lease_seconds
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE;")
            except (sqlite3.Error, OSError) as exc:
                log.warning("claim BEGIN IMMEDIATE failed: %s", exc)
                return None
            try:
                row = self._claim_locked(worker_id, now, lease_until)
                self._conn.execute("COMMIT;")
                return row
            except (sqlite3.Error, OSError) as exc:
                log.warning("claim failed for worker_id=%r: %s", worker_id, exc)
                self._safe_rollback()
                return None

    def _claim_locked(
        self, worker_id: str, now: float, lease_until: float
    ) -> Optional[Tuple[str, str]]:
        """Flip the oldest queued row to claimed. Runs inside BEGIN IMMEDIATE."""
        if self._use_returning:
            cur = self._conn.execute(
                "UPDATE jobs SET status=?, worker_id=?, lease_until=?, updated_at=? "
                "WHERE task_id = ("
                "  SELECT task_id FROM jobs WHERE status=? "
                "  ORDER BY created_at ASC, task_id ASC LIMIT 1"
                ") RETURNING task_id, payload",
                (STATUS_CLAIMED, worker_id, lease_until, now, STATUS_QUEUED),
            )
            fetched = cur.fetchone()
            if fetched is None:
                return None
            return (fetched[0], fetched[1])

        # Fallback for sqlite < 3.35 (no RETURNING). Still atomic: the whole
        # thing runs inside the same BEGIN IMMEDIATE write transaction.
        sel = self._conn.execute(
            "SELECT task_id, payload FROM jobs WHERE status=? "
            "ORDER BY created_at ASC, task_id ASC LIMIT 1",
            (STATUS_QUEUED,),
        ).fetchone()
        if sel is None:
            return None
        task_id, payload = sel[0], sel[1]
        self._conn.execute(
            "UPDATE jobs SET status=?, worker_id=?, lease_until=?, updated_at=? "
            "WHERE task_id=? AND status=?",
            (STATUS_CLAIMED, worker_id, lease_until, now, task_id, STATUS_QUEUED),
        )
        return (task_id, payload)

    def complete(self, task_id: str, result: str) -> None:
        """Mark a claimed job done and store its result text.

        Only transitions a job that is currently 'claimed' (guards against
        completing a job the watchdog already requeued). A no-match is logged,
        not raised.
        """
        self._finalize(task_id, STATUS_DONE, result_col=result, reason_col=None)

    def park(self, task_id: str, reason: str) -> None:
        """Mark a job parked for a human, recording why (the harness trail).

        Parking is allowed from either 'claimed' or 'queued' — a human decision
        to shelve a job should not be blocked by a lost lease.
        """
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "UPDATE jobs SET status=?, reason=?, updated_at=? "
                    "WHERE task_id=? AND status IN (?, ?)",
                    (
                        STATUS_PARKED,
                        reason,
                        now,
                        task_id,
                        STATUS_CLAIMED,
                        STATUS_QUEUED,
                    ),
                )
                if cur.rowcount == 0:
                    log.warning(
                        "park no-op: task_id=%r not in claimed/queued state",
                        task_id,
                    )
            except (sqlite3.Error, OSError) as exc:
                log.warning("park failed for task_id=%r: %s", task_id, exc)

    def _finalize(
        self,
        task_id: str,
        new_status: str,
        *,
        result_col: Optional[str],
        reason_col: Optional[str],
    ) -> None:
        """Shared terminal transition from 'claimed' -> done. Lock-guarded."""
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "UPDATE jobs SET status=?, result=?, reason=?, updated_at=? "
                    "WHERE task_id=? AND status=?",
                    (new_status, result_col, reason_col, now, task_id, STATUS_CLAIMED),
                )
                if cur.rowcount == 0:
                    log.warning(
                        "complete/finalize no-op: task_id=%r not in 'claimed' "
                        "state (already done, parked, or requeued?)",
                        task_id,
                    )
            except (sqlite3.Error, OSError) as exc:
                log.warning("finalize failed for task_id=%r: %s", task_id, exc)

    def watchdog_requeue(self) -> int:
        """Requeue every claimed job whose lease has expired. Returns the count.

        Crash-recovery: a worker that died mid-job leaves its claim to expire,
        and this pass returns it to 'queued' (clearing the worker/lease stamp)
        so another worker can pick it up. Returns 0 on any sqlite error.
        """
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "UPDATE jobs SET status=?, worker_id=NULL, lease_until=NULL, "
                    "updated_at=? WHERE status=? AND lease_until IS NOT NULL "
                    "AND lease_until < ?",
                    (STATUS_QUEUED, now, STATUS_CLAIMED, now),
                )
                count = cur.rowcount if cur.rowcount is not None else 0
                if count:
                    log.info("watchdog requeued %d expired-lease job(s)", count)
                return count
            except (sqlite3.Error, OSError) as exc:
                log.warning("watchdog_requeue failed: %s", exc)
                return 0

    # -- read helpers (ops visibility; not part of frozen surface) -----------

    def status_of(self, task_id: str) -> Optional[str]:
        """Return the current status string for ``task_id``, or None if absent."""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT status FROM jobs WHERE task_id=?", (task_id,)
                ).fetchone()
                return row[0] if row is not None else None
            except (sqlite3.Error, OSError) as exc:
                log.warning("status_of failed for task_id=%r: %s", task_id, exc)
                return None

    def counts(self) -> dict:
        """Return {status: count} across the queue (empty dict on error)."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                ).fetchall()
                return {row[0]: row[1] for row in rows}
            except (sqlite3.Error, OSError) as exc:
                log.warning("counts failed: %s", exc)
                return {}

    def _safe_rollback(self) -> None:
        """Roll back the current transaction, swallowing rollback errors."""
        try:
            self._conn.execute("ROLLBACK;")
        except sqlite3.Error as exc:
            log.warning("rollback failed: %s", exc)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                log.warning("close failed for %s: %s", self._db_path, exc)

    # -- context manager convenience ----------------------------------------

    def __enter__(self) -> "JobQueue":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def new_worker_id(prefix: str = "worker") -> str:
    """Generate a unique worker id (helper for callers spawning workers)."""
    return "%s-%s" % (prefix, uuid.uuid4().hex[:12])
