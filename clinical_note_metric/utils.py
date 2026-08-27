"""Shared utilities."""

from __future__ import annotations

import logging
import re
from typing import Iterable

from clinical_note_metric.models import SECTIONS

LOGGER = logging.getLogger("clinical_note_metric")


SECTION_ALIASES = {
    "problem list": "Problem List",
    "problems": "Problem List",
    "subjective": "Subjective",
    "history": "Subjective",
    "objective": "Objective",
    "exam": "Objective",
    "assessment": "Assessment",
    "impression": "Assessment",
    "plan": "Plan",
}

EMPTY_SECTION_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "not applicable",
    "no applicable",
    "nil",
    "-",
}


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic comparisons."""

    if not text:
        return ""
    normalized = text.lower().strip()
    normalized = re.sub(r"(\w+)-(\w+)", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 4)


def is_empty_section_text(text: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z0-9/ ]", "", text).lower().strip()
    return cleaned in EMPTY_SECTION_VALUES


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like clinical fragments."""

    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s*", text)
    return [part.strip(" -\t\r\n.") for part in parts if part.strip(" -\t\r\n.")]


def parse_sections(note: str) -> dict[str, str]:
    """Parse SOAP-style sections, tolerating missing headers."""

    sections = {section: "" for section in SECTIONS}
    if not note or is_empty_section_text(note):
        return sections

    header_re = re.compile(
        r"(?im)^\s*(problem list|problems|subjective|history|objective|exam|assessment|impression|plan)\s*:\s*"
    )
    matches = list(header_re.finditer(note))
    if not matches:
        sections["Subjective"] = note.strip()
        return sections

    for index, match in enumerate(matches):
        alias = normalize_text(match.group(1))
        section = SECTION_ALIASES[alias]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(note)
        sections[section] = (sections[section] + "\n" + note[start:end].strip()).strip()
    return sections


def safe_mean(values: Iterable[float], default: float = 100.0) -> float:
    values = list(values)
    if not values:
        return default
    return sum(values) / len(values)

