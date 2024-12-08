from __future__ import annotations
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class MecabMatrix:
    data: list[list[str]] = field(default_factory=list)