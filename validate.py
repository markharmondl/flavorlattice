"""Validation gate.

Every record extracted from unstructured sources passes through here before it
can reach the canonical store. The checks are all *physical or structural* —
they test whether a record could be true, not whether a model was confident.

This distinction matters. Model-reported confidence is uncalibrated and
correlates poorly with correctness on table extraction specifically: a model
reading a well-formatted table with the wrong column header will be confident
and wrong. Unit errors, impossible retention indices, and concentrations that
exceed the mass of the sample are mechanically detectable and catch most of
what actually goes wrong.

Order of operations in the pipeline:

    extract.py  ->  staging table  ->  validate.py  ->  canonical store
                                            |
                                            +-> rejected (kept, with reason)

Rejected records are retained, not deleted. They are your evaluation set when
you swap extraction models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .schema import Compound, CompoundClass, CookingMethod, Occurrence

# Kovats retention indices on a standard nonpolar column. Anything outside this
# is either a different column type or a misread.
_RI_MIN, _RI_MAX = 400.0, 3000.0

# Plausible RI window by class, nonpolar column. Wide on purpose — this catches
# a compound assigned to the wrong family, not fine errors.
_RI_WINDOW: dict[CompoundClass, tuple[float, float]] = {
    CompoundClass.TERPENE: (900.0, 1700.0),
    CompoundClass.PYRAZINE: (800.0, 1500.0),
    CompoundClass.FURAN: (700.0, 1400.0),
    CompoundClass.ALDEHYDE: (500.0, 1600.0),
    CompoundClass.ESTER: (600.0, 1800.0),
    CompoundClass.LACTONE: (1000.0, 2100.0),
    CompoundClass.PHENOL: (950.0, 1900.0),
}

# 1e6 ppm = 100% of sample mass. A single volatile above ~10,000 ppm (1%) in a
# whole food is essentially always a unit error (mg/kg read as g/kg, or a
# headspace peak-area percentage misread as a concentration).
_CONC_HARD_MAX_PPM = 1_000_000.0
_CONC_SOFT_MAX_PPM = 10_000.0


@dataclass
class Finding:
    level: str        # "reject" | "warn"
    code: str
    detail: str


@dataclass
class ValidationResult:
    ok: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def rejections(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "reject"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]


def validate_compound(c: Compound) -> ValidationResult:
    f: list[Finding] = []

    if not c.has_stable_identity:
        f.append(Finding("reject", "no_identity",
                         f"{c.name!r} resolved to neither CID, CAS, nor InChIKey"))

    if c.odor_threshold_ppb is not None:
        if c.odor_threshold_ppb <= 0:
            f.append(Finding("reject", "threshold_nonpositive",
                             f"threshold {c.odor_threshold_ppb}"))
        elif c.odor_threshold_ppb > 1e7:
            # 10 ppm+ detection threshold — possible but rare; usually a unit slip
            f.append(Finding("warn", "threshold_implausible_high",
                             f"threshold {c.odor_threshold_ppb} ppb"))

    if c.kovats_ri is not None:
        if not (_RI_MIN <= c.kovats_ri <= _RI_MAX):
            f.append(Finding("reject", "ri_out_of_range",
                             f"RI {c.kovats_ri} outside [{_RI_MIN},{_RI_MAX}]"))
        else:
            window = _RI_WINDOW.get(c.compound_class)
            if window and not (window[0] <= c.kovats_ri <= window[1]):
                f.append(Finding("warn", "ri_class_mismatch",
                                 f"RI {c.kovats_ri} unusual for {c.compound_class.value}"))

    return ValidationResult(ok=not any(x.level == "reject" for x in f), findings=f)


def validate_occurrence(o: Occurrence, known_ingredient: bool = True,
                        known_compound: bool = True) -> ValidationResult:
    f: list[Finding] = []

    if not known_ingredient:
        f.append(Finding("reject", "unknown_ingredient", o.ingredient_id))
    if not known_compound:
        f.append(Finding("reject", "unknown_compound", o.compound_key))
    if o.compound_key.startswith("name:"):
        f.append(Finding("reject", "unresolved_compound_key",
                         "occurrence keyed on a name, not a registry identifier"))

    lo, hi = o.concentration_ppm_min, o.concentration_ppm_max
    for label, v in (("min", lo), ("max", hi)):
        if v is None:
            continue
        if v < 0:
            f.append(Finding("reject", "concentration_negative", f"{label}={v}"))
        elif v > _CONC_HARD_MAX_PPM:
            f.append(Finding("reject", "concentration_exceeds_sample",
                             f"{label}={v} ppm > 100% of sample"))
        elif v > _CONC_SOFT_MAX_PPM:
            f.append(Finding("warn", "concentration_high",
                             f"{label}={v} ppm (>1%) — check for a unit error"))
    if lo is not None and hi is not None and lo > hi:
        f.append(Finding("reject", "range_inverted", f"min {lo} > max {hi}"))

    return ValidationResult(ok=not any(x.level == "reject" for x in f), findings=f)


def validate_cooking_delta(
    raw_ppm: Optional[float],
    cooked_ppm: Optional[float],
    method: CookingMethod,
) -> ValidationResult:
    """Sanity-check an extracted raw->cooked pair.

    The asymmetry check is the useful one: wet methods below 100 C cannot
    generate Maillard products, so a paper reporting a large pyrazine increase
    under boiling is either mislabeled or was misread. That is a real and common
    extraction error, because papers often tabulate several methods side by side
    and the column-to-method mapping is easy to shift by one.
    """
    f: list[Finding] = []
    if raw_ppm is None or cooked_ppm is None:
        return ValidationResult(ok=True, findings=f)
    if raw_ppm < 0 or cooked_ppm < 0:
        f.append(Finding("reject", "negative_concentration", ""))
        return ValidationResult(False, f)

    if raw_ppm > 0:
        ratio = cooked_ppm / raw_ppm
        if ratio > 1e4:
            f.append(Finding("warn", "delta_extreme",
                             f"{ratio:.0f}x increase — verify units match between columns"))
        if method.is_wet and ratio > 50:
            f.append(Finding("warn", "wet_method_large_gain",
                             "large gain under a wet method — check method/column alignment"))

    return ValidationResult(ok=not any(x.level == "reject" for x in f), findings=f)


def partition(
    records: Iterable[tuple[object, ValidationResult]]
) -> tuple[list[object], list[tuple[object, list[Finding]]]]:
    """Split validated records into (promotable, rejected-with-reasons)."""
    good, bad = [], []
    for rec, res in records:
        if res.ok:
            good.append(rec)
        else:
            bad.append((rec, res.rejections))
    return good, bad
