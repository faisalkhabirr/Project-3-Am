"""Deterministic case-to-supply conversion for the operational layer.

The Phase 2 onsite dataset ships three supply targets alongside the Phase 1
case series. We verified that each target is a deterministic function of the
reported weekly cases (100% row match on all 31,320 rows):

    saline_bags_needed     = round(31/13 * cases)        ~= 2.385 bags / case
    diagnostic_kits_needed = round_half_down(7/6 * cases) ~= 1.167 kits / case
    hospital_beds_required = round_half_down(0.3 * cases)  = 30% admission rate

``round`` is conventional half-up for saline; the kits and beds columns use
half-down at exact ties, which is why the epsilon is applied. For fractional
forecasts the same functions are evaluated directly, so demand scales smoothly
with the predicted case count.

Everything here is pure Python/NumPy: no Streamlit, no network, no I/O. The
allocation, routing, simulation, and dispatch modules all consume this single
source of truth so the operational numbers stay consistent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# Verified generator rates. Keep as fractions so the arithmetic is reproducible.
SALINE_PER_CASE = 31 / 13
KITS_PER_CASE = 7 / 6
BEDS_PER_CASE = 0.3

SUPPLY_COLUMNS = (
    "saline_bags_needed",
    "diagnostic_kits_needed",
    "hospital_beds_required",
)

# Exact-tie behaviour for the two half-down columns. The tolerance is far below
# any meaningful epidemiological difference (1e-9 of a bag) but safely absorbs
# IEEE-754 representation error at x.5 boundaries.
_TIE_EPS = 1e-9


@dataclass(frozen=True)
class Supplies:
    """Typed view of one ward-week's required supply."""

    saline_bags_needed: int
    diagnostic_kits_needed: int
    hospital_beds_required: int

    def as_dict(self) -> dict[str, int]:
        return {
            "saline_bags_needed": self.saline_bags_needed,
            "diagnostic_kits_needed": self.diagnostic_kits_needed,
            "hospital_beds_required": self.hospital_beds_required,
        }


def _validate_cases(cases: float) -> float:
    value = float(cases)
    if not math.isfinite(value):
        raise ValueError(f"cases must be finite, got {cases!r}")
    if value < 0:
        raise ValueError(f"cases must be non-negative, got {cases!r}")
    return value


def _round_half_down(value: float) -> int:
    """Round to the nearest integer, resolving exact ties downward."""
    # A tiny subtraction makes an exact .5 land just below the boundary. This
    # mirrors the dataset generator (e.g. 5 cases -> 1 bed, not 2).
    return int(math.floor(value + 0.5 - _TIE_EPS))


def cases_to_supplies(cases: float) -> dict[str, int]:
    """Convert predicted/reported cases into the three required supplies.

    Parameters
    ----------
    cases:
        Non-negative weekly case count. Fractional model forecasts are allowed;
        the demand then scales continuously before rounding.

    Returns
    -------
    dict with ``saline_bags_needed``, ``diagnostic_kits_needed`` and
    ``hospital_beds_required`` as Python ints.
    """
    value = _validate_cases(cases)
    return Supplies(
        saline_bags_needed=int(math.floor(value * SALINE_PER_CASE + 0.5)),
        diagnostic_kits_needed=_round_half_down(value * KITS_PER_CASE),
        hospital_beds_required=_round_half_down(value * BEDS_PER_CASE),
    ).as_dict()


def supplies_for_cases(cases: Iterable[float] | pd.Series | np.ndarray) -> pd.DataFrame:
    """Vectorized supply conversion, preserving the input index.

    This is what the command-center tables use: one pass over all 60 wards
    instead of a Python loop with repeated model reloads.
    """
    if isinstance(cases, pd.Series):
        values = pd.to_numeric(cases, errors="raise").astype(float)
        index = cases.index
    else:
        values = pd.Series(np.asarray(list(cases), dtype=float))
        index = values.index

    if values.isna().any():
        raise ValueError("cases contain NaN values")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("cases contain non-finite values")
    if (values < 0).any():
        raise ValueError("cases must be non-negative")

    saline = np.floor(values.to_numpy() * SALINE_PER_CASE + 0.5).astype(int)
    kits = np.floor(values.to_numpy() * KITS_PER_CASE + 0.5 - _TIE_EPS).astype(int)
    beds = np.floor(values.to_numpy() * BEDS_PER_CASE + 0.5 - _TIE_EPS).astype(int)

    return pd.DataFrame(
        {
            "saline_bags_needed": saline,
            "diagnostic_kits_needed": kits,
            "hospital_beds_required": beds,
        },
        index=index,
    )
