"""Cooking-method aroma transform.

There is no public dataset of "compound delta by cooking method" — the knowledge
sits in scattered GC-MS review papers. So v0 encodes the chemistry as explicit
parameterized operators over compound *families* rather than pretending to look
values up. This is deliberately legible: every number below is a knob you can
calibrate against a paper, and the whole module is meant to be replaced by
`LearnedCookingTransform` once you have paired raw/cooked GC-MS data.

The operators:

  Maillard          amino acids + reducing sugars under dry heat. Generates
                    pyrazines (roasted/nutty), furans (caramel/bready),
                    thiophenes and pyrroles. Onset near 140 C, superlinear in
                    temperature, saturating in time.
  Lipid oxidation   requires fat. Generates aldehydes (hexanal green, nonanal
                    fatty), 1-octen-3-ol (mushroom). Grows with heat; very long
                    cooking pushes past pleasant into rancid.
  Volatile loss     terpenes and other light top notes decay under ALL heat,
                    fastest in open dry heat, and leach into the water in wet
                    methods.
  Wet attenuation   below ~100 C there is effectively no Maillard, and soluble
                    volatiles migrate out of the food.

Everything is multiplicative on profile weights, so the transform composes and
never produces negative weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

from ..data.schema import AromaProfile, Compound, CompoundClass, CookingMethod

# Nominal surface temperature (C) by method, used when the caller does not
# specify one. Roasting an item at "200 C oven" does not put the surface at
# 200 C; these are effective reaction-zone temperatures.
_NOMINAL_TEMP_C: Mapping[CookingMethod, float] = {
    CookingMethod.RAW: 20.0,
    CookingMethod.SOUS_VIDE: 60.0,
    CookingMethod.POACH: 80.0,
    CookingMethod.STEAM: 100.0,
    CookingMethod.BOIL: 100.0,
    CookingMethod.BAKE: 150.0,
    CookingMethod.SMOKE: 110.0,
    CookingMethod.SAUTE: 175.0,
    CookingMethod.ROAST: 185.0,
    CookingMethod.FRY: 190.0,
    CookingMethod.GRILL: 230.0,
}

_MAILLARD_ONSET_C = 140.0
_GENERATED_FLOOR = 0.15  # weight given to a family member generated from ~zero


@dataclass
class CookingContext:
    method: CookingMethod
    minutes: float = 20.0
    temp_c: Optional[float] = None
    has_fat: bool = False
    has_protein: bool = False
    has_reducing_sugar: bool = True

    @property
    def temperature(self) -> float:
        return self.temp_c if self.temp_c is not None else _NOMINAL_TEMP_C[self.method]


class CookingTransform(Protocol):
    """Swap-in point for the learned model. Same signature both ways."""

    def apply(
        self,
        profile: AromaProfile,
        ctx: CookingContext,
        compounds: Mapping[str, Compound],
    ) -> AromaProfile: ...


def _maillard_extent(ctx: CookingContext) -> float:
    """0..1. Superlinear in temperature above onset, saturating in time."""
    if not ctx.method.is_dry_high_heat or not ctx.has_reducing_sugar:
        return 0.0
    over = ctx.temperature - _MAILLARD_ONSET_C
    if over <= 0:
        return 0.0
    thermal = min(1.0, (over / 60.0) ** 1.5)
    temporal = 1.0 - math.exp(-ctx.minutes / 15.0)
    return thermal * temporal


def _lipid_extent(ctx: CookingContext) -> float:
    """0..1, then penalized for excessive duration (rancid territory)."""
    if not ctx.has_fat or ctx.method == CookingMethod.RAW:
        return 0.0
    thermal = min(1.0, max(0.0, (ctx.temperature - 60.0) / 130.0))
    temporal = 1.0 - math.exp(-ctx.minutes / 25.0)
    return thermal * temporal


def _volatile_loss(ctx: CookingContext) -> float:
    """0..1 fraction of light top-note weight lost."""
    if ctx.method == CookingMethod.RAW:
        return 0.0
    base = min(0.85, max(0.0, (ctx.temperature - 40.0) / 200.0))
    duration = 1.0 - math.exp(-ctx.minutes / 20.0)
    leach = 0.25 if ctx.method.is_wet else 0.0
    return min(0.95, base * duration + leach)


class ChemicalCookingTransform:
    """The v0 rule-based transform. Implements `CookingTransform`."""

    def apply(
        self,
        profile: AromaProfile,
        ctx: CookingContext,
        compounds: Mapping[str, Compound],
    ) -> AromaProfile:
        maillard = _maillard_extent(ctx)
        lipid = _lipid_extent(ctx)
        loss = _volatile_loss(ctx)

        out: dict[str, float] = {}
        for key, w in profile.weights.items():
            compound = compounds.get(key)
            cls = compound.compound_class if compound else CompoundClass.OTHER
            new_w = w

            # Top notes decay under any heat. Terpenes and esters are the most
            # volatile families and go first; this is why cooked fruit stops
            # smelling like raw fruit.
            if cls in (CompoundClass.TERPENE, CompoundClass.ESTER):
                new_w *= (1.0 - loss)
            elif cls == CompoundClass.ALCOHOL:
                new_w *= (1.0 - 0.6 * loss)
            else:
                new_w *= (1.0 - 0.3 * loss)

            # Maillard products accumulate.
            if cls.is_maillard_product and maillard > 0:
                new_w = max(new_w, _GENERATED_FLOOR) * (1.0 + 4.0 * maillard)

            # Strecker aldehydes need protein; lipid aldehydes need fat.
            if cls == CompoundClass.ALDEHYDE:
                gain = 0.0
                if lipid > 0:
                    gain += 2.5 * lipid
                if maillard > 0 and ctx.has_protein:
                    gain += 2.0 * maillard
                if gain > 0:
                    new_w = max(new_w, _GENERATED_FLOOR) * (1.0 + gain)

            # Thiols are generated by both pathways but are also destroyed by
            # prolonged heat — net effect is a hump, not a ramp.
            if cls == CompoundClass.THIOL and maillard > 0:
                new_w *= (1.0 + 1.5 * maillard) * math.exp(-ctx.minutes / 90.0)

            if new_w > 1e-9:
                out[key] = new_w

        return AromaProfile(
            ingredient_id=profile.ingredient_id,
            weights=out,
            weighting=profile.weighting,
            cooking_method=ctx.method,
        )


def apply_cooking(
    profile: AromaProfile,
    ctx: CookingContext,
    compounds: Mapping[str, Compound],
    transform: Optional[CookingTransform] = None,
) -> AromaProfile:
    """Public entry point. Pass `transform` to override the rule-based default
    with a learned one; the signature is identical so callers never change."""
    return (transform or ChemicalCookingTransform()).apply(profile, ctx, compounds)


class LearnedCookingTransform:  # pragma: no cover - stub
    """Trained on raw/cooked GC-MS deltas + RecipeDB process labels.

    Wiring:
      1. Assemble training pairs from the literature-extraction pipeline
         (data/extract.py) — each is (compound_key, method, temp, minutes,
         log-ratio cooked/raw).
      2. Fit per-compound-class regressors, or a single model with class as a
         feature. Class-level pooling matters: most compounds have one or two
         observations, so a per-compound model will not fit.
      3. Predict multiplicative log-deltas, exponentiate, apply as above.

    Until then `apply` raises rather than silently returning the rule output —
    a learned transform that quietly falls back is worse than no learned
    transform, because you cannot tell which one produced a result.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def apply(
        self,
        profile: AromaProfile,
        ctx: CookingContext,
        compounds: Mapping[str, Compound],
    ) -> AromaProfile:
        raise NotImplementedError(
            "Train the transform first — see the docstring. "
            "Use ChemicalCookingTransform for the rule-based v0."
        )
