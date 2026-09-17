"""Pytest bootstrap: put the repo root on sys.path so `import grok_fleet` works
without an editable install. Kept intentionally tiny and dependency-free."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
