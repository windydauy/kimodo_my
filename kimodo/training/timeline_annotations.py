# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for loading BONES-SEED timeline annotations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class TimelineEvent:
    """Single atomic timeline segment."""

    start_time: float
    end_time: float
    description: str


@dataclass(frozen=True)
class _TimelineRecord:
    """Internal representation of one motion's timeline annotation."""

    filename: str
    overview_description: str
    events: Tuple[TimelineEvent, ...]
    propagated_from_filename: Optional[str]


def normalize_clip_name(path_or_name: PathLike) -> str:
    """Normalize a csv path or clip name to the timeline key format.

    Examples:
    - ``/x/y/jump_001__A001.csv`` -> ``jump_001__A001``
    - ``jump_001__A001`` -> ``jump_001__A001``
    """
    return Path(path_or_name).stem


class TimelineAnnotationIndex:
    """Index over ``timelines.jsonl`` keyed by BONES-SEED clip filename."""

    def __init__(self, records: Dict[str, _TimelineRecord]):
        self._records = records

    @classmethod
    def from_jsonl(cls, jsonl_path: PathLike) -> "TimelineAnnotationIndex":
        """Load timeline annotations from a JSONL file."""
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"Timeline annotation file not found: {path}")

        records: Dict[str, _TimelineRecord] = {}
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                filename = obj["filename"]
                events = tuple(
                    TimelineEvent(
                        start_time=float(e["start_time"]),
                        end_time=float(e["end_time"]),
                        description=str(e["description"]),
                    )
                    for e in obj.get("events", [])
                )
                records[filename] = _TimelineRecord(
                    filename=filename,
                    overview_description=str(obj.get("overview_description", "")),
                    events=events,
                    propagated_from_filename=obj.get("propagated_from_filename"),
                )

                if not records[filename].overview_description and not records[filename].events:
                    raise ValueError(
                        f"Invalid empty record at {path}:{line_no} for filename={filename!r}: "
                        "missing both overview_description and events."
                    )
        return cls(records=records)

    def __len__(self) -> int:
        return len(self._records)

    def has(self, clip_path_or_name: PathLike, allow_mirror_fallback: bool = True) -> bool:
        """Return whether annotation exists for clip name/path."""
        key = normalize_clip_name(clip_path_or_name)
        if key in self._records:
            return True
        if allow_mirror_fallback and key.endswith("_M"):
            return key[:-2] in self._records
        return False

    def get_record(self, clip_path_or_name: PathLike, allow_mirror_fallback: bool = True) -> Dict:
        """Return normalized annotation record dict for the clip.

        Raises:
            KeyError: if no record can be found.
        """
        key = normalize_clip_name(clip_path_or_name)
        rec = self._records.get(key)
        if rec is None and allow_mirror_fallback and key.endswith("_M"):
            rec = self._records.get(key[:-2])
        if rec is None:
            raise KeyError(f"No timeline annotation found for clip {key!r}")
        return {
            "filename": rec.filename,
            "overview_description": rec.overview_description,
            "events": [
                {"start_time": e.start_time, "end_time": e.end_time, "description": e.description}
                for e in rec.events
            ],
            "propagated_from_filename": rec.propagated_from_filename,
        }

    def sample_text(
        self,
        clip_path_or_name: PathLike,
        mode: str = "mixed",
        rng: Optional[random.Random] = None,
    ) -> str:
        """Sample a text conditioning string for a clip.

        Args:
            clip_path_or_name: CSV path or clip stem.
            mode: One of ``overview``, ``event``, or ``mixed``.
            rng: Optional Python random generator.
        """
        if mode not in {"overview", "event", "mixed"}:
            raise ValueError(f"Unknown mode={mode!r}, expected one of: overview, event, mixed")
        if rng is None:
            rng = random

        rec = self.get_record(clip_path_or_name)
        overview = rec["overview_description"].strip()
        events: Sequence[Dict] = rec["events"]

        if mode == "overview":
            if overview:
                return overview
            if events:
                return str(events[0]["description"])
            raise ValueError(f"No text available for clip={normalize_clip_name(clip_path_or_name)!r}")

        if mode == "event":
            if events:
                return str(rng.choice(events)["description"])
            if overview:
                return overview
            raise ValueError(f"No text available for clip={normalize_clip_name(clip_path_or_name)!r}")

        # mixed
        if events and overview:
            if rng.random() < 0.5:
                return overview
            return str(rng.choice(events)["description"])
        if events:
            return str(rng.choice(events)["description"])
        if overview:
            return overview
        raise ValueError(f"No text available for clip={normalize_clip_name(clip_path_or_name)!r}")

    def missing_clips(
        self,
        clip_paths_or_names: Sequence[PathLike],
        allow_mirror_fallback: bool = True,
    ) -> List[str]:
        """Return clip keys with no annotation."""
        missing = []
        for p in clip_paths_or_names:
            key = normalize_clip_name(p)
            if not self.has(key, allow_mirror_fallback=allow_mirror_fallback):
                missing.append(key)
        return missing
