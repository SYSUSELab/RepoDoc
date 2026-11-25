"""Small dependency-free progress reporter for long CLI operations.

Progress is written to stderr and flushed after every line. This matters when
the CLI is redirected to a log file: Python's default stdout buffering can
otherwise make a long-running command look completely silent.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO


class Progress:
    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        quiet: bool = False,
    ) -> None:
        self.stream = stream or sys.stderr
        self.quiet = quiet
        self.started_at = time.monotonic()

    def _elapsed(self) -> str:
        seconds = int(time.monotonic() - self.started_at)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def line(self, message: str) -> None:
        if self.quiet:
            return
        print(f"[repodoc +{self._elapsed()}] {message}", file=self.stream, flush=True)

    def stage(self, message: str) -> None:
        self.line(f"STAGE {message}")

    def step(
        self,
        label: str,
        current: int,
        total: int,
        *,
        detail: str = "",
    ) -> None:
        suffix = f" — {detail}" if detail else ""
        self.line(f"{label} [{current}/{total}]{suffix}")

    def done(
        self,
        label: str,
        current: int,
        total: int,
        *,
        detail: str = "",
    ) -> None:
        suffix = f" — {detail}" if detail else ""
        self.line(f"DONE {label} [{current}/{total}]{suffix}")

    def info(self, message: str) -> None:
        self.line(f"INFO {message}")
