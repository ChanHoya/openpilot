"""Entrypoint for `python3 -m openpilot.tools.drive_analyzer`."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
  raise SystemExit(main())
