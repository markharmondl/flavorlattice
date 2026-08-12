"""Core data model.

Deliberately storage-agnostic: these are plain dataclasses. The store layer
(store.py) is responsible for persisting them (DuckDB by default). Keeping the
domain model free of ORM decorators means the pairing engine can be unit-tested
without a database.

Compound identity is keyed on PubChem CID when available, falling back to a
normalized CAS number, then to InChIKey. FlavorDB2, FooDB and Flavornet all
carry at least one of these, so cross-source joins resolve on a stable key
rather than on names (which are noisy: "cis-3-hexenol" vs "(Z)-3-hexen-1-ol").

Identity resolution is DETERMINISTIC and lives in resolve.py. Nothing in this
package may assign a CID by inference; see the note in extract.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class CompoundClass(str, Enum):
    """Coarse functional/aroma family. Drives cooking transforms (cooking.py)
    and lets the engine reason about compound groups when concentration data
    is missing. Assigned during ingest from functional group + SMILES."""

    PYRAZINE = "pyrazine"          # roasted, nutty — Maillard marker
    FURAN = "furan"                # caramel, bready — Maillard marker
    THIOPHENE = "thiophene"        # meaty, savory — Maillard marker
    PYRROLE = "pyrrole"            # cereal, roasted
    ALDEHYDE = "aldehyde"          # green (hexanal) / fatty (nonanal) / malty
    KETONE = "ketone"              # buttery, creamy
    ALCOHOL = "alcohol"            # green, fusel, mushroom (1-octen-3-ol)
    ESTER = "ester"                # fruity — dominant in raw fruit
    TERPENE = "terpene"            # citrus, pine, herbal — volatile top notes
    LACTONE = "lactone"            # coconut, peach, creamy
    THIOL = "thiol"                # alliaceous, tropical, meaty; very low threshold
    SULFIDE = "sulfide"            # brassica, allium
    PHENOL = "phenol"              # smoky, clove, medicinal
    ACID = "acid"                  # sour, cheesy, rancid
    OTHER = "other"

    @property
    def is_maillard_product(self) -> bool:
        return self in {
            CompoundClass.PYRAZINE,
            CompoundClass.FURAN,
            CompoundClass.THIOPHENE,
            CompoundClass.PYRROLE,
        }


class CookingMethod(str, Enum):
    """Processes the cooking transform understands. Maps onto (a subset of)
    RecipeDB's 268 process labels — see data/recipedb_process_map.py when you
    wire that source."""

    RAW = "raw"
    BOIL = "boil"
    STEAM = "steam"
    POACH = "poach"
    SOUS_VIDE = "sous_vide"
    SAUTE = "saute"
    FRY = "fry"
    BAKE = "bake"
    ROAST = "roast"
    GRILL = "grill"
    SMOKE = "smoke"

    @property
    def is_wet(self) -> bool:
        return self in {
            CookingMethod.BOIL,
            CookingMethod.STEAM,
            CookingMethod.POACH,
            CookingMethod.SOUS_VIDE,
        }

    @property
    def is_dry_high_heat(self) -> bool:
        return self in {
            CookingMethod.SAUTE,
            CookingMethod.FRY,
            CookingMethod.BAKE,
            CookingMethod.ROAST,
            CookingMethod.GRILL,
            CookingMethod.SMOKE,
        }


_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def normalize_cas(raw: Optional[str]) -> Optional[str]:
    """CAS numbers arrive with stray whitespace, 'CAS ' prefixes, and unicode
    dashes. Normalize before using as an identity fallback."""
    if not raw:
        return None
    s = str(raw).strip().upper().replace("CAS", "").strip()
    s = s.replace("\u2010", "-").replace("\u2013", "-").replace("\u2212", "-")
    s = s.replace(" ", "")
    return s if _CAS_RE.match(s) else None


def normalize_inchikey(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().upper()
    return s if _INCHIKEY_RE.match(s) else None


@dataclass(frozen=True)
class Provenance:
    """Attached to every fact that came from outside the repo. Required for
    anything promoted out of staging — see staging.py."""

    source: str                          # "flavordb2" | "foodb" | "flavornet" | "literature"
    locator: Optional[str] = None        # URL, DOI, or accession
    retrieved_at: Optional[str] = None   # ISO 8601
    page: Optional[int] = None           # for literature extractions


@dataclass
class Compound:
    """A volatile (or tastant) molecule.

    `odor_threshold_ppb` is the orthonasal detection threshold. It is what turns
    a concentration into an odor activity value (OAV = concentration/threshold),
    which is far closer to perceptual weight than raw concentration. Missing
    thresholds are common; vectorize.py falls back to class medians.
    """

    name: str
    cid: Optional[int] = None                 # PubChem CID — preferred identity
    cas: Optional[str] = None
    inchikey: Optional[str] = None
    smiles: Optional[str] = None
    compound_class: CompoundClass = CompoundClass.OTHER
    odor_descriptors: tuple[str, ...] = ()
    odor_threshold_ppb: Optional[float] = None
    kovats_ri: Optional[float] = None         # linear retention index; volatility axis
    fema: Optional[str] = None
    provenance: Optional[Provenance] = None

    def __post_init__(self) -> None:
        self.cas = normalize_cas(self.cas)
        self.inchikey = normalize_inchikey(self.inchikey)
        if isinstance(self.odor_descriptors, list):
            self.odor_descriptors = tuple(self.odor_descriptors)

    @property
    def key(self) -> str:
        """Stable cross-source join key.

        Precedence is deliberate: CID is a curated identity that already merges
        synonyms; CAS is registry-stable but occasionally assigned per-isomer;
        InChIKey is structure-derived and distinguishes stereoisomers (which we
        WANT — cis-3-hexenol and trans-3-hexenol smell different). Name is the
        last resort and is flagged so it can be audited.
        """
        if self.cid is not None:
            return f"cid:{self.cid}"
        if self.cas:
            return f"cas:{self.cas}"
        if self.inchikey:
            return f"ikey:{self.inchikey}"
        return f"name:{self.name.strip().lower()}"

    @property
    def has_stable_identity(self) -> bool:
        return not self.key.startswith("name:")


@dataclass
class Occurrence:
    """A compound observed in an ingredient, optionally quantified.

    Concentrations are stored as a range because FooDB reports literature ranges,
    not point values. vectorize.py uses the geometric mean of the range when both
    bounds exist — concentration data is log-distributed, so the arithmetic mean
    overweights the upper bound.
    """

    ingredient_id: str
    compound_key: str
    concentration_ppm_min: Optional[float] = None
    concentration_ppm_max: Optional[float] = None
    cooking_method: CookingMethod = CookingMethod.RAW
    provenance: Optional[Provenance] = None

    @property
    def concentration_ppm(self) -> Optional[float]:
        lo, hi = self.concentration_ppm_min, self.concentration_ppm_max
        if lo is not None and hi is not None and lo > 0 and hi > 0:
            return (lo * hi) ** 0.5
        return lo if lo is not None else hi


@dataclass
class Ingredient:
    id: str                              # slug: "sweet_basil"
    name: str
    category: Optional[str] = None       # FlavorDB2 category (34 of them)
    aliases: tuple[str, ...] = ()
    contains_fat: bool = False           # gates the lipid-oxidation operator
    contains_reducing_sugar: bool = True # gates Maillard
    contains_protein: bool = False       # gates Strecker
    provenance: Optional[Provenance] = None

    def __post_init__(self) -> None:
        if isinstance(self.aliases, list):
            self.aliases = tuple(self.aliases)


@dataclass
class AromaProfile:
    """The engine's unit of work: an ingredient reduced to weighted compounds.

    `weights` maps compound_key -> non-negative weight. Under BINARY weighting
    every present compound is 1.0; under OAV weighting it is log1p(OAV). The
    profile records which, because you cannot meaningfully compare a binary
    profile against an OAV one.
    """

    ingredient_id: str
    weights: dict[str, float] = field(default_factory=dict)
    weighting: str = "binary"            # "binary" | "oav"
    cooking_method: CookingMethod = CookingMethod.RAW

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda kv: -kv[1])[:n]

    def compounds(self) -> Iterable[str]:
        return self.weights.keys()
