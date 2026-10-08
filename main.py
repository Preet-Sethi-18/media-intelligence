"""Compatibility entry point; prefer `uv run media-intelligence`."""

from media_intelligence.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
