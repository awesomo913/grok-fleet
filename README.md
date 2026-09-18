# Grok Fleet

> A stdlib-only Python harness that forces multiple AI models to do real work, prove it, and clear a 5-gate quality bar before a result is trusted.

Grok Fleet (package name `grok_fleet`) is a review/routing harness for orchestrating multiple AI models on the same task: it routes work, runs each result through contract, anti-simulation, rubric, verification, and review gates, and trips a circuit breaker around models/providers that keep failing.

## Features
- **Router** (`router.py`) assigns work across models; **queue** (`queue.py`) manages distribution.
- **Circuit breaker** isolates a failing model/provider instead of letting it keep eating work.
- **5-gate quality bar**: contract (`contract.py`), anti-simulation (`antisim.py`), rubric (`rubric.py`), verify (`verify.py`), and review (`review.py`) all have to pass.
- **Typed interfaces** (`interfaces.py`, `types.py`) keep the harness pluggable across model backends.
- Zero runtime dependencies — pure Python 3.9+ standard library, only `threading` from stdlib for concurrency.
- Full pytest suite with roughly 1:1 module-to-test coverage (`tests/`).

## Stack
Python 3.9+, standard library only (no runtime dependencies); `pytest` for the test suite.

## Getting started
**Requirements**
- Python 3.9+

**Run**
```bash
pip install -e .
python -c "import grok_fleet"   # or use the harness/router/review modules directly
pip install -e ".[dev]" && pytest   # to run the test suite
```

## Status
**Unmaintained / archived.** Personal project, published as-is — fork it, adapt it, take it over. No support or guarantees.

## License
[MIT](LICENSE) — free to use, fork, and build on.
