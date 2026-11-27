"""Operational engines for the Phase 2 command center.

Pure, deterministic, offline-safe modules:

- ``supplies``   case -> saline / kits / beds conversion (single source of truth)
- ``allocation`` scarcity-aware resource allocation
- ``routing``    intra-city hospital overflow routing
- ``simulation`` model-driven vector-control counterfactuals
"""
from .supplies import (
    BEDS_PER_CASE,
    KITS_PER_CASE,
    SALINE_PER_CASE,
    SUPPLY_COLUMNS,
    Supplies,
    cases_to_supplies,
    supplies_for_cases,
)

__all__ = [
    "BEDS_PER_CASE",
    "KITS_PER_CASE",
    "SALINE_PER_CASE",
    "SUPPLY_COLUMNS",
    "Supplies",
    "cases_to_supplies",
    "supplies_for_cases",
]
