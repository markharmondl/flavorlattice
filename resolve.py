"""Compound identity resolution. DETERMINISTIC ONLY.

This is the single most failure-prone step in the whole system and the one
place where a language model must never be in the loop.

Why: every downstream number — overlap, cosine, OAV — is computed over compound
keys. If two distinct molecules get merged under one key, or one molecule gets
split across two, the error does not surface as an obvious bug. It surfaces as
subtly wrong pairings that look plausible. You will not catch it by inspection.

The specific hazards:

  Stereoisomers      (Z)-3-hexen-1-ol smells of cut grass; (E)-3-hexen-1-ol is
                     weaker and different. They share a molecular formula and
                     nearly share a name. A fuzzy matcher merges them.
  Positional isomers 2-methylpyrazine vs 2,3-dimethylpyrazine — different odor,
                     different threshold by an order of magnitude.
  Synonym sprawl     one compound may carry a dozen trade/trivial names across
                     FEMA, CAS registries, and papers.
  Hallucinated IDs   an LLM asked for "the PubChem CID of hexanal" will produce
                     a plausible integer whether or not it is correct.

So: names go in, registry lookups come out, and anything that fails to resolve
is quarantined rather than guessed at.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from urllib.request import urlopen

from .schema import normalize_cas, normalize_inchikey

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DEFAULT_CACHE = Path("data_store/resolve_cache.json")


@dataclass(frozen=True)
class Resolution:
    query: str
    cid: Optional[int]
    inchikey: Optional[str]
    canonical_name: Optional[str]
    method: str      # "cache" | "cas" | "inchikey" | "name" | "unresolved"

    @property
    def ok(self) -> bool:
        return self.cid is not None


class CompoundResolver:
    """Resolves names/CAS/InChIKey to a PubChem CID, with an on-disk cache.

    Cache the results. PubChem rate-limits at roughly 5 requests/second and an
    initial FlavorDB2 ingest is ~25k molecules; without a cache you will spend
    hours re-resolving the same compounds on every rerun.
    """

    def __init__(
        self,
        cache_path: Path | str = DEFAULT_CACHE,
        offline: bool = False,
        min_interval_s: float = 0.21,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.offline = offline
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_call
        if delta < self.min_interval_s:
            time.sleep(self.min_interval_s - delta)
        self._last_call = time.monotonic()

    def _fetch_cid(self, namespace: str, value: str) -> Optional[int]:
        if self.offline:
            return None
        self._throttle()
        url = f"{PUBCHEM_BASE}/compound/{namespace}/{quote(value)}/cids/JSON"
        try:
            with urlopen(url, timeout=15) as r:  # noqa: S310 - fixed host
                payload = json.load(r)
            cids = payload.get("IdentifierList", {}).get("CID", [])
            # More than one CID means the identifier is ambiguous. Refuse it
            # rather than taking the first — "take the first" is exactly how
            # isomers get merged.
            return int(cids[0]) if len(cids) == 1 else None
        except Exception:
            return None

    def resolve(
        self,
        name: str,
        cas: Optional[str] = None,
        inchikey: Optional[str] = None,
    ) -> Resolution:
        ck = f"{name}|{cas or ''}|{inchikey or ''}".lower()
        if ck in self._cache:
            c = self._cache[ck]
            return Resolution(name, c.get("cid"), c.get("inchikey"),
                              c.get("canonical_name"), "cache")

        cas_n, ikey_n = normalize_cas(cas), normalize_inchikey(inchikey)
        cid, method = None, "unresolved"

        # Precedence: structure-derived first, then registry, then name. Name
        # lookup is last because it is the only one that can be ambiguous in a
        # way PubChem will happily resolve to the wrong tautomer.
        if ikey_n:
            cid, method = self._fetch_cid("inchikey", ikey_n), "inchikey"
        if cid is None and cas_n:
            cid, method = self._fetch_cid("name", cas_n), "cas"
        if cid is None and name:
            cid, method = self._fetch_cid("name", name.strip()), "name"

        res = Resolution(name, cid, ikey_n, name if cid else None,
                         method if cid else "unresolved")
        self._cache[ck] = {
            "cid": res.cid, "inchikey": res.inchikey,
            "canonical_name": res.canonical_name,
        }
        return res

    def flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=0))
