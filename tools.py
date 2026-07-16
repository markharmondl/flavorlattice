"""Deterministic tools the LLM orchestrator may call.

These wrap the engine so the model never does math or invents compounds -- it
only decides WHICH tool to call with WHICH args, then narrates the returned
data. Each tool is a plain function with a JSON-serializable result and an
OpenAI-style schema in TOOL_SCHEMAS for function-calling / constrained decode.
"""

from __future__ import annotations

from ..data.schema import CookingMethod
from ..engine import pairing, vectorize
from ..engine.cooking import TransformParams, apply_cooking


class Tools:
    def __init__(self, store, compounds, resolver):
        self.store = store
        self.compounds = compounds          # {cid: Compound}
        self.resolver = resolver            # name -> ing_id (embedding + alias)
        self._ings = store.load_ingredients()

    def resolve_ingredient(self, name: str) -> dict:
        ing_id = self.resolver(name)
        ing = self._ings.get(ing_id)
        return {"ing_id": ing_id, "name": ing.name if ing else None}

    def profile(self, ing_id: str, method: str = "raw", temp_c: float = 200,
                minutes: float = 20) -> dict:
        ing = self._ings[ing_id]
        vec = vectorize.ingredient_vector(ing, self.compounds)
        if method != "raw":
            klass = {cid: self.compounds[cid].klass for cid in vec if cid in self.compounds}
            vec = apply_cooking(vec, klass, CookingMethod(method),
                                TransformParams(temp_c=temp_c, minutes=minutes))
        top = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)[:15]
        return {"ing_id": ing_id, "method": method,
                "top_compounds": [{"cid": c, "weight": round(w, 3)} for c, w in top]}

    def pair(self, ing_a: str, ing_b: str, method: str = "raw") -> dict:
        va = self._vec(ing_a, method)
        vb = self._vec(ing_b, method)
        r = pairing.pair_score(ing_a, va, ing_b, vb)
        return {"shared": r.shared, "similarity": round(r.similarity, 3),
                "novelty": round(r.novelty, 3), "top_shared": r.top_shared}

    def suggest_partners(self, ing_id: str, mode: str = "harmony",
                         method: str = "raw", limit: int = 15) -> dict:
        q = self._vec(ing_id, method)
        cands = {i: vectorize.ingredient_vector(ing, self.compounds)
                 for i, ing in self._ings.items() if i != ing_id}
        results = pairing.rank_partners(q, cands, mode=mode, limit=limit)
        return {"mode": mode, "partners": [
            {"ing_id": r.b, "similarity": round(r.similarity, 3),
             "novelty": round(r.novelty, 3), "shared": r.shared} for r in results]}

    def _vec(self, ing_id: str, method: str) -> dict:
        ing = self._ings[ing_id]
        vec = vectorize.ingredient_vector(ing, self.compounds)
        if method == "raw":
            return vec
        klass = {cid: self.compounds[cid].klass for cid in vec if cid in self.compounds}
        return apply_cooking(vec, klass, CookingMethod(method))


TOOL_SCHEMAS = [
    {"name": "resolve_ingredient", "description": "fuzzy name -> canonical id",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                    "required": ["name"]}},
    {"name": "suggest_partners", "description": "best pairings for an ingredient",
     "parameters": {"type": "object", "properties": {
         "ing_id": {"type": "string"},
         "mode": {"type": "string", "enum": ["harmony", "contrast"]},
         "method": {"type": "string"}, "limit": {"type": "integer"}},
         "required": ["ing_id"]}},
    {"name": "pair", "description": "score two ingredients",
     "parameters": {"type": "object", "properties": {
         "ing_a": {"type": "string"}, "ing_b": {"type": "string"},
         "method": {"type": "string"}}, "required": ["ing_a", "ing_b"]}},
    {"name": "profile", "description": "top aroma compounds, raw or cooked",
     "parameters": {"type": "object", "properties": {
         "ing_id": {"type": "string"}, "method": {"type": "string"},
         "temp_c": {"type": "number"}, "minutes": {"type": "number"}},
         "required": ["ing_id"]}},
]