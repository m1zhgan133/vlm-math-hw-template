from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MathVQASample:
    id: str
    image: Image.Image
    question: str
    options: list[str]
    answer: str
    subject: str
    source: str = "unknown"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc

    return rows


def sanitize_question(text: str) -> str:
    for token in ("<image>", "<image_start>", "<image_end>"):
        text = text.replace(token, "")
    return " ".join(text.split())


class MathVQADataset(Dataset[MathVQASample]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        max_samples: int | None = None,
        subjects: Iterable[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.split = split
        self.max_samples = max_samples
        self.subjects = set(subjects) if subjects is not None else None

        rows = load_jsonl(self.manifest_path)

        if split is not None:
            rows = [r for r in rows if r.get("split") == split]

        if self.subjects is not None:
            rows = [r for r in rows if r.get("subject") in self.subjects]

        if max_samples is not None:
            rows = rows[: max(0, int(max_samples))]

        self.rows: list[dict[str, Any]] = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> MathVQASample:
        row = self.rows[idx]

        image_path = self.root / row["image"]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        options = list(row.get("options") or [])

        return MathVQASample(
            id=str(row["id"]),
            image=image,
            question=sanitize_question(str(row["question"])),
            options=[str(o) for o in options],
            answer=str(row["answer"]).strip(),
            subject=str(row.get("subject", "unknown")),
            source=str(row.get("source", "unknown")),
        )
