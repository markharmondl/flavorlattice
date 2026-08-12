"""Literature extraction: GC-MS tables -> staged occurrence records.

This is the ONE place in the pipeline where a language model earns its cost.
Everything else (FlavorDB2, FooDB, Flavornet, the Ahn edge list) is structured,
bulk-downloadable, and stable-schema — a hand-written adapter beats an agent on
cost, speed, reproducibility and diffability, and there are only a handful of
those sources so adapter-writing amortizes fine.

The cooking-method transform data is different. No database has it. It lives in
individual food-science papers with inconsistent table layout, inconsistent
units, inconsistent internal standards, and the cooking treatment often encoded
only in a column header or figure caption. That is genuine per-document
reasoning over unstructured input.

SHAPE OF THE PIPELINE
---------------------
Deterministic search/fetch/dedupe  ->  LLM extraction (this module)  ->  staging
->  deterministic validation  ->  canonical

The model is an extractor, not an orchestrator. It sees one document and emits
records. It does not decide what to fetch next, does not manage the queue, and
does not write to the canonical store. This keeps the nondeterministic surface
to a single, replayable, cacheable call per document.

WHAT THE MODEL MUST NOT DO
--------------------------
- Assign a PubChem CID, CAS, or any registry identifier. It emits the compound
  name AS PRINTED; resolve.py resolves it. See the hazards list in resolve.py.
- Convert units. It reports the value and the unit string as printed; conversion
  is deterministic and testable. Model-side unit conversion is a leading source
  of silent 1000x errors.
- Infer values not present in the table. An empty cell is empty, not zero.

BATCH, NOT LOCAL
----------------
Run this against a cheap API model, not the 8GB local box. This is a one-time
backfill whose accuracy propagates into every downstream score, and it is not
on the latency path — a different budget from the interactive PARSE/EXPLAIN
work in models/. DeepSeek-V3 or Qwen via the same OpenAI-compatible client in
models/api.py is the intended target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from .schema import CookingMethod
from .staging import StagedOccurrence

EXTRACTOR_VERSION = "v0.1-unimplemented"

# Unit conversions to ppm. Deterministic on purpose — see module docstring.
_UNIT_TO_PPM: dict[str, float] = {
    "ppm": 1.0,
    "mg/kg": 1.0,
    "ug/g": 1.0,
    "µg/g": 1.0,
    "ppb": 0.001,
    "ug/kg": 0.001,
    "µg/kg": 0.001,
    "ng/g": 0.001,
    "mg/100g": 10.0,
    "g/kg": 1000.0,
    "percent": 10000.0,
    "%": 10000.0,
}


def to_ppm(value: float, unit: str) -> Optional[float]:
    """Convert a reported value to ppm, or None if the unit is unrecognized.

    Returning None (rather than guessing) is intentional: an unrecognized unit
    becomes a validation rejection with a legible reason, which surfaces in the
    rejection report as a fixable pattern.

    Note the deliberate omission of peak-area percentages ("relative %", "area
    %"). Those are NOT concentrations and must not be coerced into ppm — a
    large fraction of GC-MS papers report only relative abundance, and treating
    that as concentration is the single most common way this corpus goes wrong.
    """
    key = unit.strip().lower().replace(" ", "")
    return value * _UNIT_TO_PPM[key] if key in _UNIT_TO_PPM else None


EXTRACTION_PROMPT = """\
You are extracting volatile compound measurements from a food science paper.

Return ONLY a JSON array. No preamble, no markdown fences, no commentary.

Each element:
{
  "compound_name":   string,  // EXACTLY as printed in the table. Do not normalize,
                              // expand, or correct it. Keep isomer prefixes
                              // ((Z)-, (E)-, cis-, trans-, 2-, 3-) verbatim.
  "ingredient_name": string,  // the food, as the paper names it
  "value_min":       number|null,
  "value_max":       number|null,  // if a single value is given, put it in both
  "unit_as_printed": string,  // verbatim: "mg/kg", "ug/g", "relative %", etc.
  "cooking_method":  string|null,  // one of: raw, boil, steam, poach, sous_vide,
                                   // saute, fry, bake, roast, grill, smoke
  "table_ref":       string|null,  // e.g. "Table 2"
  "raw_span":        string   // the verbatim row text you read this from
}

Rules:
- Do NOT assign PubChem CIDs, CAS numbers, or any registry identifier.
- Do NOT convert units. Report the number and unit exactly as printed.
- An empty or "n.d." cell means the measurement is absent. Use null, never 0.
- If the cooking method for a column is ambiguous, set cooking_method to null.
  A null is recoverable; a wrong method label silently corrupts the transform.
- If the table reports relative peak area rather than concentration, still
  report it with unit_as_printed set to what is printed. Downstream validation
  will handle it.
"""


class ExtractionClient(Protocol):
    """Implemented by models/api.py. Kept as a Protocol so extraction can be
    unit-tested with a canned client and no network."""

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str: ...


@dataclass
class Document:
    doi: str
    text: str
    page: Optional[int] = None


def _staging_id(doi: str, compound: str, ingredient: str, method: str) -> str:
    h = hashlib.sha256(f"{doi}|{compound}|{ingredient}|{method}".encode()).hexdigest()
    return h[:24]


def extract_document(
    doc: Document,
    client: ExtractionClient,
    extractor_version: str = EXTRACTOR_VERSION,
) -> list[StagedOccurrence]:
    """Extract staged records from one document.

    Wiring (not yet implemented):
      1. client.complete(EXTRACTION_PROMPT, doc.text) -> JSON array
      2. json.loads with a fence-stripping fallback (models emit ```json despite
         instructions; strip rather than fail)
      3. For each element: to_ppm(value, unit_as_printed). Unrecognized unit ->
         leave concentrations None and let validation reject it with a reason.
      4. Build StagedOccurrence with raw_span PRESERVED VERBATIM.
      5. Return; caller stages them. compound_key stays None here — resolve.py
         fills it in a separate deterministic pass.

    Note step 5. Resolution is a separate pass so that a resolution failure and
    an extraction failure appear as distinct rejection codes. Collapsing them
    makes the rejection report useless for deciding what to fix.
    """
    raise NotImplementedError(
        "Wire an ExtractionClient from models/api.py, then implement steps 1-5 "
        "in the docstring. Everything downstream (staging, validation, promotion) "
        "is implemented and tested."
    )


def parse_extraction_response(payload: str) -> list[dict]:
    """Tolerant JSON parse. Models emit fences despite instructions."""
    s = payload.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found in extraction response")
    return json.loads(s[start : end + 1])


def to_staged(
    rows: Sequence[dict],
    doc: Document,
    extractor_version: str = EXTRACTOR_VERSION,
) -> list[StagedOccurrence]:
    """Deterministic conversion of parsed rows into staging records."""
    out: list[StagedOccurrence] = []
    for row in rows:
        unit = row.get("unit_as_printed") or ""
        vmin, vmax = row.get("value_min"), row.get("value_max")
        ppm_min = to_ppm(vmin, unit) if isinstance(vmin, (int, float)) else None
        ppm_max = to_ppm(vmax, unit) if isinstance(vmax, (int, float)) else None

        raw_method = (row.get("cooking_method") or "").strip().lower()
        try:
            method = CookingMethod(raw_method) if raw_method else CookingMethod.RAW
        except ValueError:
            method = CookingMethod.RAW

        out.append(
            StagedOccurrence(
                staging_id=_staging_id(
                    doc.doi, row.get("compound_name", ""),
                    row.get("ingredient_name", ""), method.value,
                ),
                ingredient_name=row.get("ingredient_name", ""),
                compound_name=row.get("compound_name", ""),
                raw_span=row.get("raw_span", ""),
                extractor_version=extractor_version,
                compound_key=None,          # resolve.py fills this
                concentration_ppm_min=ppm_min,
                concentration_ppm_max=ppm_max,
                unit_as_reported=unit,
                cooking_method=method,
                doi=doc.doi,
                page=doc.page,
                table_ref=row.get("table_ref"),
            )
        )
    return out
