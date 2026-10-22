"""Anti-Hallucination Numeric Grounding Verifier.

Extracts all numbers from the simulation tool outputs and verifies that numerical
claims in the agent's explanation match computed values within rounding tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, List, Set


# Standard citation / structural numbers that do not require simulation-output matching
EXEMPT_CONSTANTS = {
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    12.0,
    13.0,
    2000.0,
    2001.0,
    2009.0,
    2011.0,
    2014.0,
    2015.0,
    2016.0,
    2017.0,
}


@dataclass
class GroundingReport:
    passed: bool
    verified_numbers: List[float] = field(default_factory=list)
    unverified_numbers: List[float] = field(default_factory=list)
    tool_number_pool_size: int = 0

    @property
    def summary_badge(self) -> str:
        total = len(self.verified_numbers) + len(self.unverified_numbers)
        if total == 0:
            return "✓ Grounded in SR Engine (qualitative summary)"
        if self.passed:
            return f"✓ Grounded: {len(self.verified_numbers)}/{total} numerical claims verified against SR Engine"
        return (
            f"⚠ Grounding Check: {len(self.verified_numbers)}/{total} numbers verified "
            f"(unmatched: {self.unverified_numbers[:3]})"
        )


def collect_numbers_from_obj(obj: Any, out: Set[float]) -> None:
    """Recursively collect all ints and floats from a nested dict/list/tuple tool output."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        val = float(obj)
        if not (val != val):  # skip NaN
            out.add(val)
            # Also add percentage equivalent (e.g. 0.90 -> 90.0)
            if 0.0 < abs(val) <= 1.0:
                out.add(round(val * 100.0, 2))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, (int, float)):
                out.add(float(k))
            collect_numbers_from_obj(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            collect_numbers_from_obj(item, out)


def verify_response_numbers(
    response_text: str,
    tool_outputs: Any,
    rel_tol: float = 0.03,
    abs_tol: float = 0.05,
) -> GroundingReport:
    """Check that numerical claims in `response_text` appear in `tool_outputs`."""
    pool: Set[float] = set()
    collect_numbers_from_obj(tool_outputs, pool)

    # Strip figure/equation/preset labels (e.g., Fig. 3b, Eq. 7, P4, 1D, 2D, U_1, #12) before extracting numbers
    cleaned = re.sub(
        r"\b(?:Fig(?:ure)?s?|Supp(?:lementary)?|Eq(?:uation)?s?|Preset|P\d+|1D|2D|3D|U_\d+|V_\d+|M_\d+|s\d+|#\d+)\s*[\d.–\-]*[a-z]?",
        " ",
        response_text,
        flags=re.IGNORECASE,
    )
    # Strip markdown headers and numbered list prefixes at start of lines
    cleaned = re.sub(r"(?m)^\s*\d+\.\s+", " ", cleaned)

    tokens = re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", cleaned)
    verified: List[float] = []
    unverified: List[float] = []

    for tok in tokens:
        try:
            val = float(tok)
        except ValueError:
            continue
        if val in EXEMPT_CONSTANTS:
            continue

        matched = False
        for ref in pool:
            if abs(val - ref) <= max(abs_tol, rel_tol * max(1.0, abs(ref))):
                matched = True
                break
        if matched:
            verified.append(val)
        else:
            unverified.append(val)

    return GroundingReport(
        passed=(len(unverified) == 0),
        verified_numbers=verified,
        unverified_numbers=unverified,
        tool_number_pool_size=len(pool),
    )
