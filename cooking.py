"""Cooking transforms: how a raw aroma profile changes under a cooking method.

This is the part with no clean public dataset. There is no unified DB of
"compound delta by cooking method"; the knowledge lives scattered across GC-MS
review papers. But the chemistry is systematic enough to encode as parameterized
operators, which is what this module does. Two well-characterized pathways
dominate thermal processing:

  Maillard reaction (amino acids + reducing sugars, dry heat >~140C):
      -> pyrazines (roasted/nutty), furans (caramel/bready), thiophenes/thiols
         (meaty), Strecker aldehydes (malty/cocoa). Grows with BOTH temperature
         and time.

  Lipid oxidation (free fatty acid -> carbonyls):
      -> aldehydes (hexanal green, octanal/nonanal fatty), 2-pentylfuran,
         1-octen-3-ol (mushroom). Grows with temperature, but excessive TIME
         degrades it. Requires fat in the ingredient.

Wet methods (boil/steam, capped at ~100C) suppress Maillard almost entirely and
strip volatiles into the cooking water, so they mostly ATTENUATE the raw profile
rather than add to it. Terpenes and other heat-labile top notes decay under all
heat.

IMPORTANT: this is a v0 heuristic operator, intentionally simple and legible.
It is exactly the component to replace with a learned model once you have
paired raw/cooked GC-MS data (RecipeDB's 268 process labels + per-ingredient
GC-MS studies are the training signal). See models/embeddings.py for where a
learned CookingTransform would slot in with the same interface.

The operator works on CompoundClass groups, not individual CIDs, because the
generated Maillard/oxidation products are a known family; ingest tags each
compound with a class, and a small catalog of representative product compounds
(REPRESENTATIVES) is what actually gets injected so the output profile stays in
the same {cid: weight} space the pairing engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.schema import CompoundClass, CookingMethod


@dataclass
class TransformParams:
    temp_c: float = 200.0
    minutes: float = 20.0
    has_fat: bool = True         # gate lipid-oxidation pathway
    has_sugar: bool = True       # gate Maillard/caramelization


# Per-method operator: multipliers applied to existing compounds by class, plus
# which product pathways to inject. Multipliers <1 attenuate, >1 amplify.
# These numbers are placeholders to be calibrated against GC-MS deltas — they
# encode direction and rough magnitude, not measured values.
_METHOD_OPS: dict[CookingMethod, dict] = {
    CookingMethod.RAW: dict(retain=1.0, maillard=0.0, lipidox=0.0, terpene=1.0),
    CookingMethod.BOIL: dict(retain=0.45, maillard=0.0, lipidox=0.1, terpene=0.5),
    CookingMethod.STEAM: dict(retain=0.6, maillard=0.0, lipidox=0.1, terpene=0.6),
    CookingMethod.BAKE: dict(retain=0.8, maillard=0.5, lipidox=0.4, terpene=0.6),
    CookingMethod.ROAST: dict(retain=0.8, maillard=1.0, lipidox=0.7, terpene=0.4),
    CookingMethod.GRILL: dict(retain=0.75, maillard=1.0, lipidox=0.8, terpene=0.4),
    CookingMethod.FRY: dict(retain=0.85, maillard=0.9, lipidox=1.0, terpene=0.5),
    CookingMethod.SEAR: dict(retain=0.9, maillard=1.1, lipidox=0.8, terpene=0.5),
    CookingMethod.CARAMELIZE: dict(retain=0.7, maillard=0.6, lipidox=0.2, terpene=0.3),
    CookingMethod.SMOKE: dict(retain=0.9, maillard=0.4, lipidox=0.3, terpene=0.6),
    CookingMethod.FERMENT: dict(retain=1.0, maillard=0.0, lipidox=0.0, terpene=0.9),
}

# Classes attenuated together as "volatile top notes"
_VOLATILE_CLASSES = {CompoundClass.TERPENE, CompoundClass.ESTER}

# Representative product compounds injected per pathway. Real cids/thresholds get
# filled from FlavorDB2 at ingest; placeholders here keep the module runnable.
# (cid, class, base_weight)
REPRESENTATIVES = {
    "maillard": [
        ("cid:pyrazine_2ethyl35dimethyl", CompoundClass.PYRAZINE, 1.0),
        ("cid:2acetylpyrazine", CompoundClass.PYRAZINE, 0.8),
        ("cid:furfural", CompoundClass.FURAN, 0.7),
        ("cid:2methyl3furanthiol", CompoundClass.THIOL, 0.9),
        ("cid:3methylbutanal", CompoundClass.STRECKER_ALDEHYDE, 0.8),
    ],
    "lipidox": [
        ("cid:hexanal", CompoundClass.LIPID_ALDEHYDE, 1.0),
        ("cid:nonanal", CompoundClass.LIPID_ALDEHYDE, 0.7),
        ("cid:2pentylfuran", CompoundClass.FURAN, 0.6),
        ("cid:1octen3ol", CompoundClass.ALCOHOL, 0.5),
    ],
    "smoke": [
        ("cid:guaiacol", CompoundClass.PHENOL, 1.0),
        ("cid:4methylguaiacol", CompoundClass.PHENOL, 0.7),
    ],
    "ferment": [
        ("cid:ethyl_acetate", CompoundClass.ESTER, 0.8),
        ("cid:ethyl_lactate", CompoundClass.ESTER, 0.6),
    ],
}


def _severity(temp_c: float, minutes: float) -> float:
    """Dimensionless 0..~1.5 thermal severity. Saturating in both temp and
    time so absurd inputs don't blow up. Calibrate against real kinetics later."""
    import math
    t = max(0.0, (temp_c - 100.0) / 150.0)      # 100C->0, 250C->1
    m = 1.0 - math.exp(-minutes / 20.0)          # saturates over ~1hr
    return min(1.5, t * (0.5 + 0.5 * m))


def apply_cooking(
    raw_vec: dict[str, float],
    compound_class: dict[str, CompoundClass],
    method: CookingMethod,
    params: TransformParams | None = None,
) -> dict[str, float]:
    """Return a new {cid: weight} profile for the cooked ingredient.

    `compound_class` maps every cid in raw_vec to its CompoundClass (built at
    ingest). Unmapped cids are treated as OTHER and only get the global retain
    multiplier.
    """
    params = params or TransformParams()
    op = _METHOD_OPS[method]
    sev = _severity(params.temp_c, params.minutes)

    out: dict[str, float] = {}

    # 1. attenuate/retain existing compounds
    for cid, w in raw_vec.items():
        klass = compound_class.get(cid, CompoundClass.OTHER)
        factor = op["retain"]
        if klass in _VOLATILE_CLASSES:
            factor *= op["terpene"]
        out[cid] = w * factor

    # 2. inject Maillard products
    if op["maillard"] > 0 and params.has_sugar:
        for cid, _klass, base in REPRESENTATIVES["maillard"]:
            out[cid] = out.get(cid, 0.0) + base * op["maillard"] * sev

    # 3. inject lipid-oxidation products (time can over-degrade -> mild penalty)
    if op["lipidox"] > 0 and params.has_fat:
        time_penalty = 1.0 if params.minutes <= 30 else 0.8
        for cid, _klass, base in REPRESENTATIVES["lipidox"]:
            out[cid] = out.get(cid, 0.0) + base * op["lipidox"] * sev * time_penalty

    # 4. method-specific extras
    if method == CookingMethod.SMOKE:
        for cid, _klass, base in REPRESENTATIVES["smoke"]:
            out[cid] = out.get(cid, 0.0) + base
    if method == CookingMethod.FERMENT:
        for cid, _klass, base in REPRESENTATIVES["ferment"]:
            out[cid] = out.get(cid, 0.0) + base

    return {cid: w for cid, w in out.items() if w > 1e-6}