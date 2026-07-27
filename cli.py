#!/usr/bin/env python
"""Root CLI shim so `python cli.py ...` works (CLAUDE.md §0.6, §4).

The real Typer app lives in ``heron/cli.py``.
"""

from heron.cli import app

if __name__ == "__main__":
    app()
