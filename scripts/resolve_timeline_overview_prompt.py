#!/usr/bin/env python3
"""Resolve a single-prompt text from timeline annotations for one clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kimodo.training.timeline_annotations import TimelineAnnotationIndex, normalize_clip_name


def resolve_overview_prompt(
    *,
    timeline_jsonl: str | Path,
    clip: str | Path,
    fallback: str | None = None,
) -> tuple[str, str]:
    """Resolve prompt text from timeline overview, then event[0], then fallback.

    Returns:
        (prompt, source), where source is one of:
        - "overview"
        - "event"
        - "fallback"
    """
    index = TimelineAnnotationIndex.from_jsonl(timeline_jsonl)
    try:
        record = index.get_record(clip)
    except KeyError:
        if fallback is not None and str(fallback).strip():
            return str(fallback).strip(), "fallback"
        raise

    overview = str(record.get("overview_description", "")).strip()
    if overview:
        return overview, "overview"

    for event in record.get("events", []):
        text = str(event.get("description", "")).strip()
        if text:
            return text, "event"

    if fallback is not None and str(fallback).strip():
        return str(fallback).strip(), "fallback"

    clip_key = normalize_clip_name(clip)
    raise ValueError(
        f"No overview/event text available for clip={clip_key!r}, and no fallback prompt was provided."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve one overview prompt for a clip from timeline JSONL."
    )
    parser.add_argument("--timeline_jsonl", required=True, help="Path to timeline JSONL.")
    parser.add_argument("--clip", required=True, help="Clip path or clip stem.")
    parser.add_argument("--fallback", default=None, help="Fallback prompt if no timeline text is found.")
    parser.add_argument(
        "--print_source",
        action="store_true",
        help="Print the selected source type to stderr for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt, source = resolve_overview_prompt(
        timeline_jsonl=args.timeline_jsonl,
        clip=args.clip,
        fallback=args.fallback,
    )
    if args.print_source:
        print(f"prompt_source={source}", file=sys.stderr)
    print(prompt)


if __name__ == "__main__":
    main()
