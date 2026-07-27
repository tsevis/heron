"""Heron — a physically-grounded radiometric imaging engine.

The package is deliberately UI-agnostic: nothing under ``heron/`` imports a web
framework or any UXP/desktop shell (CLAUDE.md §2.5). Frontends call the engine;
the engine calls nothing above it.
"""

__version__ = "0.0.1"
