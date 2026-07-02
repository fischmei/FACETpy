"""Shared label normalization helpers for FACETpy CLI modules."""

from __future__ import annotations

import re


def _compact_label(value: str, fallback: str = "recording") -> str:
    """Convert free-form text into a compact alphanumeric label."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value)
    return cleaned or fallback
