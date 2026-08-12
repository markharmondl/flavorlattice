"""Staging tier and the promotion job.

The write barrier. Extraction (LLM) writes here; a deterministic job validates
and promotes into canonical. Nothing else crosses.

Design points worth keeping:

1. The RAW SPAN IS PERSISTED alongside the structured record. When you swap in
   a better extraction model in six months, you re-extract from `raw_span`
   rather than re-crawling every paper. Re-fetching is the expensive, rate-
   limited, link-rot-prone part; keep it a one-time cost.

2. Records carry `extractor_version`. Two records disagreeing about the same
   (doi, table, compound) is normal across model versions and is signal, not
   corruption — the promotion job resolves conflicts by version precedence and
   logs the disagreement.

3. Rejected records are RETAINED with their reasons. That set is your eval
   harness: when you change the extraction prompt or model, the metric that
   matters is how many previously-rejected records now pass validation, and how
   many previously-passing records now fail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

from .schema import CookingMethod, Occurrence, Provenance
from .store import Store
from .validate import ValidationResult, validate_occurrence

STAGING_SQL = """
CREATE TABLE IF NOT EXISTS staged_occurrences (
    staging_id         TEXT PRIMARY KEY,
    ingredient_name    TEXT NOT NULL,   -- as written in the paper, pre-resolution
    compound_name      TEXT NOT NULL,   -- as written in the paper
    compound_key       TEXT,            -- filled by resolve.py, NULL until resolved
    concentration_ppm_min DOUBLE,
    concentration_ppm_max DOUBLE,
    unit_as_reported   TEXT,
    cooking_method     TEXT,
    doi                TEXT,
    page               INTEGER,
    table_ref          TEXT,
    raw_span           TEXT NOT NULL,   -- verbatim source text; never discard
    extractor_version  TEXT NOT NULL,
    extracted_at       TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'pending',  -- pending|promoted|rejected
    findings           TEXT             -- JSON list of validation findings
);

CREATE INDEX IF NOT EXISTS staged_status ON staged_occurrences (status);
CREATE INDEX IF NOT EXISTS staged_doi    ON staged_occurrences (doi);
"""


@dataclass
class StagedOccurrence:
    staging_id: str
    ingredient_name: str
    compound_name: str
    raw_span: str
    extractor_version: str
    compound_key: Optional[str] = None
    concentration_ppm_min: Optional[float] = None
    concentration_ppm_max: Optional[float] = None
    unit_as_reported: Optional[str] = None
    cooking_method: CookingMethod = CookingMethod.RAW
    doi: Optional[str] = None
    page: Optional[int] = None
    table_ref: Optional[str] = None
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "pending"
    findings: list[dict] = field(default_factory=list)


class StagingStore:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.con = store.con
        self.con.execute(STAGING_SQL)

    def stage(self, records: Iterable[StagedOccurrence]) -> int:
        rows = [
            (
                r.staging_id, r.ingredient_name, r.compound_name, r.compound_key,
                r.concentration_ppm_min, r.concentration_ppm_max, r.unit_as_reported,
                r.cooking_method.value, r.doi, r.page, r.table_ref, r.raw_span,
                r.extractor_version, r.extracted_at, r.status, json.dumps(r.findings),
            )
            for r in records
        ]
        if not rows:
            return 0
        self.con.executemany(
            "INSERT OR REPLACE INTO staged_occurrences VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        return len(rows)

    def pending(self, limit: int = 1000) -> list[StagedOccurrence]:
        rows = self.con.execute(
            "SELECT staging_id,ingredient_name,compound_name,compound_key,"
            "concentration_ppm_min,concentration_ppm_max,unit_as_reported,"
            "cooking_method,doi,page,table_ref,raw_span,extractor_version,"
            "extracted_at,status FROM staged_occurrences "
            "WHERE status = 'pending' LIMIT ?", [limit]
        ).fetchall()
        return [
            StagedOccurrence(
                staging_id=r[0], ingredient_name=r[1], compound_name=r[2],
                compound_key=r[3], concentration_ppm_min=r[4],
                concentration_ppm_max=r[5], unit_as_reported=r[6],
                cooking_method=CookingMethod(r[7]), doi=r[8], page=r[9],
                table_ref=r[10], raw_span=r[11], extractor_version=r[12],
                extracted_at=r[13], status=r[14],
            )
            for r in rows
        ]

    def mark(self, staging_id: str, status: str, findings: list[dict]) -> None:
        self.con.execute(
            "UPDATE staged_occurrences SET status = ?, findings = ? WHERE staging_id = ?",
            [status, json.dumps(findings), staging_id],
        )

    def rejection_report(self) -> list[tuple[str, int]]:
        """Rejection codes by frequency. This is the list to work down when
        improving extraction — the top code is usually a systematic prompt or
        parsing problem, not a hard case."""
        rows = self.con.execute(
            "SELECT findings FROM staged_occurrences WHERE status = 'rejected'"
        ).fetchall()
        counts: dict[str, int] = {}
        for (blob,) in rows:
            for f in json.loads(blob or "[]"):
                if f.get("level") == "reject":
                    counts[f["code"]] = counts.get(f["code"], 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])


def promote(
    staging: StagingStore,
    ingredient_ids: set[str],
    compound_keys: set[str],
    batch: int = 1000,
) -> dict[str, int]:
    """Validate pending records and move the good ones into canonical.

    Deliberately NOT resolving identities here — records must already carry a
    resolved `compound_key` from resolve.py. A record arriving with a NULL key
    is rejected, not resolved on the fly, so that resolution failures show up in
    the rejection report instead of being silently retried inside promotion.
    """
    promoted = rejected = 0
    to_write: list[Occurrence] = []

    for rec in staging.pending(limit=batch):
        if not rec.compound_key:
            staging.mark(rec.staging_id, "rejected",
                         [{"level": "reject", "code": "unresolved_identity",
                           "detail": f"{rec.compound_name!r} has no compound_key"}])
            rejected += 1
            continue

        occ = Occurrence(
            ingredient_id=rec.ingredient_name,
            compound_key=rec.compound_key,
            concentration_ppm_min=rec.concentration_ppm_min,
            concentration_ppm_max=rec.concentration_ppm_max,
            cooking_method=rec.cooking_method,
            provenance=Provenance(
                source="literature", locator=rec.doi,
                retrieved_at=rec.extracted_at, page=rec.page,
            ),
        )
        res: ValidationResult = validate_occurrence(
            occ,
            known_ingredient=rec.ingredient_name in ingredient_ids,
            known_compound=rec.compound_key in compound_keys,
        )
        findings = [asdict(f) for f in res.findings]
        if res.ok:
            to_write.append(occ)
            staging.mark(rec.staging_id, "promoted", findings)
            promoted += 1
        else:
            staging.mark(rec.staging_id, "rejected", findings)
            rejected += 1

    if to_write:
        staging.store.upsert_occurrences(to_write)

    return {"promoted": promoted, "rejected": rejected}
