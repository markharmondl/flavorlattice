"""Pairing computation. Pure math — no LLM, no network, no I/O.

Everything here is deterministic and cheap, which is the whole point of the
architecture: the model layer never computes a pairing, it only parses the
request and narrates the answer.

TWO AXES, NOT ONE
-----------------
The food-pairing hypothesis (Ahn et al., Sci Rep 2011) is that ingredients
sharing volatile compounds pair well. It reproduces Western cuisine practice
but *inverts* for East Asian cuisines, which preferentially combine ingredients
that share FEW compounds. A single score therefore cannot be right for both, so
the engine reports:

  harmony  cosine similarity of the weighted profiles. High = large shared
           aromatic surface. This is the Western/foodpairing.com axis.

  novelty  contrast weighted by *bridgeability*: pairs that are far apart in
           compound space but still share at least a few high-weight anchor
           compounds. Pure dissimilarity is not interesting (chalk and cheese
           share nothing); dissimilarity with an anchor is where the East Asian
           pattern and most creative pairings live.

Callers pick the axis via `mode`. Neither is "the" score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional, Sequence

from ..data.schema import AromaProfile

Mode = Literal["harmony", "novelty", "balanced"]


@dataclass
class PairScore:
    a: str
    b: str
    harmony: float                        # 0..1 cosine
    novelty: float                        # 0..1
    n_shared: int
    shared_weight: float                  # summed min-weight over shared compounds
    anchors: list[tuple[str, float]] = field(default_factory=list)  # top shared
    mode: Mode = "harmony"

    @property
    def score(self) -> float:
        if self.mode == "harmony":
            return self.harmony
        if self.mode == "novelty":
            return self.novelty
        return 0.5 * (self.harmony + self.novelty)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"PairScore({self.a}~{self.b} "
            f"harmony={self.harmony:.3f} novelty={self.novelty:.3f} "
            f"shared={self.n_shared})"
        )


def _cosine(x: Mapping[str, float], y: Mapping[str, float]) -> float:
    if not x or not y:
        return 0.0
    if len(x) > len(y):
        x, y = y, x
    dot = sum(w * y[k] for k, w in x.items() if k in y)
    if dot == 0.0:
        return 0.0
    nx = math.sqrt(sum(w * w for w in x.values()))
    ny = math.sqrt(sum(w * w for w in y.values()))
    return dot / (nx * ny) if nx and ny else 0.0


def score_pair(
    pa: AromaProfile,
    pb: AromaProfile,
    mode: Mode = "harmony",
    n_anchors: int = 5,
) -> PairScore:
    """Score one pair. Profiles must share a weighting scheme."""
    if pa.weighting != pb.weighting:
        raise ValueError(
            f"cannot compare {pa.weighting!r} against {pb.weighting!r} profiles — "
            "binary and OAV weights are not on the same scale"
        )

    wa, wb = pa.weights, pb.weights
    shared = set(wa) & set(wb)
    harmony = _cosine(wa, wb)

    # Shared mass: how much of each profile's weight sits in the overlap. Using
    # min() means a compound only counts to the extent BOTH ingredients carry it.
    shared_weight = sum(min(wa[k], wb[k]) for k in shared)
    total = min(sum(wa.values()), sum(wb.values())) or 1.0
    overlap_frac = shared_weight / total

    # Novelty: distant overall, but with real anchors. The anchor term saturates
    # quickly — three strong shared compounds is enough of a bridge; more just
    # moves the pair back toward plain harmony.
    anchor_strength = 1.0 - math.exp(-3.0 * overlap_frac)
    novelty = (1.0 - harmony) * anchor_strength

    anchors = sorted(
        ((k, min(wa[k], wb[k])) for k in shared), key=lambda kv: -kv[1]
    )[:n_anchors]

    return PairScore(
        a=pa.ingredient_id,
        b=pb.ingredient_id,
        harmony=harmony,
        novelty=novelty,
        n_shared=len(shared),
        shared_weight=shared_weight,
        anchors=anchors,
        mode=mode,
    )


def partners(
    target: AromaProfile,
    candidates: Iterable[AromaProfile],
    mode: Mode = "harmony",
    limit: int = 20,
    min_shared: int = 1,
    prior: Optional[Mapping[tuple[str, str], float]] = None,
    prior_weight: float = 0.0,
) -> list[PairScore]:
    """Rank candidates against a target.

    `prior` is an optional recipe co-occurrence prior (from RecipeDB), keyed by
    sorted ingredient-id pair, in 0..1. It is applied as a *penalty* in novelty
    mode (a pair every recipe already uses is not novel) and as a mild *bonus*
    in harmony mode (real cooks corroborating the chemistry). Default weight is
    0.0 — the prior is off until you have wired RecipeDB and validated it.
    """
    out: list[PairScore] = []
    for cand in candidates:
        if cand.ingredient_id == target.ingredient_id:
            continue
        ps = score_pair(target, cand, mode=mode)
        if ps.n_shared < min_shared:
            continue
        if prior and prior_weight:
            key = tuple(sorted((ps.a, ps.b)))  # type: ignore[assignment]
            p = prior.get(key, 0.0)
            if mode == "novelty":
                ps.novelty = max(0.0, ps.novelty - prior_weight * p)
            else:
                ps.harmony = min(1.0, ps.harmony + prior_weight * p)
        out.append(ps)

    out.sort(key=lambda s: -s.score)
    return out[:limit]


def shared_compounds(pa: AromaProfile, pb: AromaProfile) -> list[str]:
    return sorted(set(pa.weights) & set(pb.weights))


class PairingModel:
    """Interface for a learned re-ranker (models/embeddings.py).

    The deterministic scorer above produces the candidate set; a learned model
    re-orders it. Keeping this as an interface means the engine stays testable
    with no model present, and a GNN can be dropped in without touching callers.
    """

    def rerank(
        self, target: AromaProfile, scored: Sequence[PairScore]
    ) -> list[PairScore]:  # pragma: no cover - interface
        raise NotImplementedError
