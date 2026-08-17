"""Validation issue and report types (no imports from sibling modules)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Issue:
    """One validation finding. Errors block promotion; warnings do not."""

    level: str  # "error" | "warning"
    code: str
    message: str
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "code": self.code, "message": self.message, "context": self.context}


@dataclass
class ValidationReport:
    """All findings for one raw artifact."""

    source: str
    issues: list[Issue] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, context: str = "") -> None:
        self.issues.append(Issue(level, code, message, context))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": "ok" if self.passed else "failed",
            "issue_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
        }