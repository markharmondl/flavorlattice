"""Turn an ingredient's compound occurrences into a numeric profile vector.

Two weighting schemes, selectable per query:

  BINARY  presence/absence over compounds. This is exactly the Barabási et al.
          (2011) representation; pairing then reduces to shared-compound count.
          Use when you have no concentration data.

  OAV     odor activity value = concentration / aroma_threshold. A compound
          present far above its detection threshold dominates the perceived
          aroma, so OAV weighting is closer to what a nose actually integrates.
          Requires concentration (FooDB / GC-MS) AND threshold (FlavorDB2/
          Flavornet). Falls back to BINARY per-compound when either is missing.

Vectors are sparse dicts {cid: weight}; the pairing layer decides the metric.
We keep them as dicts rather than dense numpy arrays because the compound
vocabulary is ~25k and any single ingredient touches <1%% of it.
"""

from __future__ import annotations

import math
from enum import Enum

from ..data.schema import Compound, Ingredient


class Weighting(str, Enum):
    BINARY = "binary"
    OAV = "oav"


def _geomean(a: float, b: float) -> float:
    return math.sqrt(a * b)


def ingredient_vector(
    ing: Ingredient,
    compounds: dict[str, Compound],
    weighting: Weighting = Weighting.OAV,
    log_oav: bool = True,
) -> dict[str, float]:
    """Return {cid: weight}. OAV weights are log1p-compressed by default so a
    single very-high-OAV compound doesn't swamp cosine similarity."""
    vec: dict[str, float] = {}
    for occ in ing.occurrences:
        if weighting is Weighting.BINARY:
            vec[occ.cid] = 1.0
            continue

        comp = compounds.get(occ.cid)
        thr = comp.aroma_threshold_ppm if comp else None
        if thr and occ.conc_ppm_min is not None and occ.conc_ppm_max is not None:
            conc = _geomean(occ.conc_ppm_min, occ.conc_ppm_max)
            oav = conc / thr
            vec[occ.cid] = math.log1p(oav) if log_oav else oav
        else:
            # missing concentration or threshold -> degrade gracefully to presence
            vec[occ.cid] = 1.0
    return vec


def l2_normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}