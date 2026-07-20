"""Pairing scores. This is pure math with no LLM in the loop — that is the
central design decision. The perceptual claim behind foodpairing.com (shared
volatile compounds => pairs well) is a deterministic computation over the
profile vectors. The LLM never scores a pair; it only parses the request and
narrates the result. That keeps the expensive/nondeterministic component off
the critical path and makes results reproducible and cacheable.

Three scorers:

  shared_compounds   raw |C_i ∩ C_j|. The Barabási metric. Interpretable,
                     needs only presence data, but biased toward compound-rich
                     ingredients (beef shares a lot with everything).

  weighted_cosine    cosine over OAV vectors. Corrects the richness bias and
                     accounts for perceptual dominance. Default when
                     concentrations are available.

  novelty            East-Asian-cuisine mode from the same paper: reward pairs
                     that DON'T share compounds. Returned as a separate axis so
                     the caller can trade off "harmony" vs "contrast".

pair_score returns both a similarity and a novelty component so downstream code
(or the user's own learned re-ranker) can combine them however it wants rather
than collapsing to one number too early.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PairResult:
    a: str
    b: str
    shared: int
    similarity: float      # weighted cosine in [0, 1]
    novelty: float         # 1 - jaccard, in [0, 1]
    top_shared: list[str]  # cids driving the score, for explanation


def _dot(u: dict[str, float], v: dict[str, float]) -> float:
    small, large = (u, v) if len(u) <= len(v) else (v, u)
    return sum(w * large.get(k, 0.0) for k, w in small.items())


def _cosine(u: dict[str, float], v: dict[str, float]) -> float:
    import math
    nu = math.sqrt(_dot(u, u))
    nv = math.sqrt(_dot(v, v))
    if nu == 0 or nv == 0:
        return 0.0
    return _dot(u, v) / (nu * nv)


def _jaccard(u: dict[str, float], v: dict[str, float]) -> float:
    su, sv = set(u), set(v)
    union = su | sv
    return (len(su & sv) / len(union)) if union else 0.0


def pair_score(
    name_a: str,
    vec_a: dict[str, float],
    name_b: str,
    vec_b: dict[str, float],
    top_k: int = 8,
) -> PairResult:
    shared_cids = set(vec_a) & set(vec_b)
    # rank shared compounds by their contribution to the dot product
    contrib = sorted(
        shared_cids, key=lambda c: vec_a[c] * vec_b[c], reverse=True
    )
    return PairResult(
        a=name_a,
        b=name_b,
        shared=len(shared_cids),
        similarity=_cosine(vec_a, vec_b),
        novelty=1.0 - _jaccard(vec_a, vec_b),
        top_shared=contrib[:top_k],
    )


def rank_partners(
    query_vec: dict[str, float],
    candidates: dict[str, dict[str, float]],
    mode: str = "harmony",
    limit: int = 20,
) -> list[PairResult]:
    """Score `query_vec` against every candidate profile and return the best.

    mode="harmony"  -> sort by similarity (classic food-pairing hypothesis)
    mode="contrast" -> sort by novelty (East-Asian-style avoidance)
    """
    results = [
        pair_score("query", query_vec, name, vec)
        for name, vec in candidates.items()
    ]
    key = (lambda r: r.similarity) if mode == "harmony" else (lambda r: r.novelty)
    results.sort(key=key, reverse=True)
    return results[:limit]