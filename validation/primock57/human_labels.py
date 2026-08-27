"""Parse the human-annotated 'Incorrect statements' / 'Omissions' cells from
primock57's results.csv. Each cell is newline-separated items; an item
prefixed with '!' is critical, one prefixed with '-' is non-critical.
"""

from __future__ import annotations


def parse_labels(text: str | None) -> tuple[int, int]:
    """Return (critical_count, noncritical_count) for one cell's text."""

    critical = 0
    noncritical = 0
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("!"):
            critical += 1
        elif line.startswith("-"):
            noncritical += 1
    return critical, noncritical
