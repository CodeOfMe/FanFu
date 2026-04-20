"""Standard result container for FanFu operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Standard result container for FanFu operations."""

    success: bool
    data: Any = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.success
