from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Granularity = Literal["file", "leaf_folder"]

CATEGORY_STORE_VERSION = 1


@dataclass
class CategoryColumn:
    """One independent CSV column (e.g. sex vs dose)."""

    id: str
    title: str
    values: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "values": list(self.values)}

    @staticmethod
    def from_json(d: dict[str, Any]) -> CategoryColumn:
        return CategoryColumn(
            id=str(d.get("id", "")).strip(),
            title=str(d.get("title", "")).strip(),
            values=[str(x).strip() for x in (d.get("values") or []) if str(x).strip()],
        )


@dataclass
class CategoryStore:
    """Assignments keyed by path relative to ``input_root`` (POSIX, no leading slash)."""

    input_root: str
    granularity: Granularity
    columns: list[CategoryColumn]
    assignments: dict[str, dict[str, str]]
    version: int = CATEGORY_STORE_VERSION

    def column_ids(self) -> list[str]:
        return [c.id for c in self.columns if c.id]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "input_root": self.input_root,
            "granularity": self.granularity,
            "columns": [c.to_json() for c in self.columns],
            "assignments": {k: dict(v) for k, v in self.assignments.items()},
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> CategoryStore:
        cols = [CategoryColumn.from_json(x) for x in (d.get("columns") or [])]
        cols = [c for c in cols if c.id]
        g = d.get("granularity", "file")
        if g not in ("file", "leaf_folder"):
            g = "file"
        raw_a = d.get("assignments") or {}
        assignments: dict[str, dict[str, str]] = {}
        if isinstance(raw_a, dict):
            for k, v in raw_a.items():
                key = str(k).strip().replace("\\", "/").lstrip("/")
                if not key:
                    continue
                if isinstance(v, dict):
                    assignments[key] = {str(ck): str(cv) for ck, cv in v.items() if str(ck).strip()}
        return CategoryStore(
            version=int(d.get("version", CATEGORY_STORE_VERSION)),
            input_root=str(d.get("input_root", "")).strip(),
            granularity=g,  # type: ignore[arg-type]
            columns=cols,
            assignments=assignments,
        )
