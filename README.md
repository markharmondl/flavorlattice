# flavorpair

A local-first flavor/aroma pairing engine in the spirit of foodpairing.com:
recommend ingredient pairings from shared volatile-compound profiles, adjusted
for cooking method. Deterministic pairing math with a thin, cost-controlled LLM
layer for the natural-language interface.

## The one design decision that matters

**The LLM does not compute pairings.** The food-pairing hypothesis — ingredients
that share volatile aroma compounds tend to pair well — is a deterministic
computation over profile vectors (`engine/`). The LLM only (1) parses a request
into a structured query and (2) narrates the result. This keeps the expensive,
nondeterministic component off the critical path: results are reproducible and
cacheable, and the system stays useful with no model installed at all (the CLI's
`pair`/`partners`/`profile` commands hit the engine directly).

That split is also what makes it cheap. Almost every token you'd otherwise spend
asking a model "what goes with X" is replaced by a table lookup and a cosine.

## Data (the hard part)

Pairing quality is bounded by the compound-profile data, not the code.

- **FlavorDB2** (cosylab, IIIT-Delhi) — primary source. 25,595 flavor molecules;
  2,254 mapped to 936 natural ingredients across 34 categories. Carries SMILES,
  CAS, FEMA, functional group, odor descriptors, and **aroma/taste threshold
  values**. Thresholds are essential: they convert concentration into odor
  activity value (OAV = concentration / threshold), which is closer to what a
  nose integrates than raw presence.
- **FooDB** (foodb.ca) — ~70k constituents with **concentration ranges** per
  food from the literature. Join on compound id to upgrade a presence/absence
  profile to OAV-weighted. Bulk dump available.
- **Flavornet** (Acree & Arn) — GC-O aroma compounds with **Kovats retention
  indices** and odor descriptors. Fills odor character and volatility.
- **Barabási flavor network** (Sci Rep 2011) — ~1,530 ingredients, ~37k
  shared-compound edges. Use as a ready baseline before FlavorDB2 ingest
  finishes, and as a validation set (your scores should reproduce their
  cuisine-level trends).
- **RecipeDB** (cosylab) — ~118k recipes tagged with **268 cooking processes**,
  linked to FlavorDB molecules. Two uses: a co-occurrence prior (pairs real
  recipes actually use) and **training labels for the learned cooking
  transform** (process label → observed profile shift).
- **VCF** (Volatile Compounds in Food, TNO) — commercial/paywalled gold standard
  for quantitative volatiles. Upgrade path, not a v0 dependency.

### How data gets in: two planes, one write barrier

The bulk sources above are structured, stable-schema and few, so they get plain
deterministic ETL (`data/ingest.py`). **No model in that loop** — a hand-written
adapter beats an agent on cost, speed, reproducibility and diffability, and at
N≈5 sources adapter-writing amortizes fine. Agentic ingestion earns its cost
when the source set is large and heterogeneous enough that adapters don't.

The **cooking-method transform data is the exception**, and it's the one place a
model earns its keep. No database has it; it's scattered across individual
GC-MS papers with inconsistent table layout, units, and internal standards, with
the treatment often encoded only in a column header. That's genuine per-document
reasoning over unstructured input.

```
  BULK PLANE     download -> parse -> validate -> canonical        (no model)

  LONG-TAIL      search/fetch/dedupe  (deterministic)
  PLANE            -> extract         (LLM, one document at a time)
                   -> staging table
                   -> resolve         (deterministic, PubChem)
                   -> validate        (deterministic, physical checks)
                   -> canonical
```

Three rules the code enforces:

1. **An LLM may write to staging; never to canonical.** `data/staging.py` is the
   barrier. Promotion is a deterministic job.
2. **Identity resolution is never inferred.** The model emits the compound name
   *as printed*; `data/resolve.py` resolves it against PubChem and refuses
   ambiguous hits rather than taking the first. An LLM-assigned CID that merges
   (Z)- and (E)-3-hexen-1-ol poisons every downstream score in a way you will
   not catch by inspection.
3. **Validation is physical, not confidence-based.** `data/validate.py` checks
   retention indices against class windows, concentrations against sample mass,
   ranges for inversion, and — the useful one — large Maillard-product gains
   under wet methods, which is the signature of a shifted method-to-column
   mapping. Model-reported confidence is uncalibrated on table extraction; a
   model reading the wrong column is confident and wrong.

Rejected records are **retained with their reasons**. That set is the eval
harness: when you change the extraction prompt or model, the metric is how many
previously-rejected records now pass, and how many previously-passing now fail.
`StagingStore.rejection_report()` ranks failure codes — the top one is almost
always a systematic prompt bug, not a hard case.

Run extraction on a **cheap API model, not the local 8GB box**. It's a one-time
batch backfill whose accuracy propagates into every score; that's a different
budget from the latency-sensitive interactive path.

### Cooking-method aroma changes

There is no clean public dataset of "compound delta by cooking method" — the
knowledge is scattered across GC-MS review papers. But the chemistry is
systematic, so `engine/cooking.py` encodes it as parameterized operators over
compound families rather than trying to look it up:

- **Maillard** (amino acids + reducing sugars, dry heat): adds pyrazines
  (roasted/nutty), furans (caramel/bready), thiophenes/thiols (meaty), Strecker
  aldehydes (malty/cocoa). Grows with temperature and time.
- **Lipid oxidation** (needs fat): adds aldehydes (hexanal green, nonanal fatty),
  2-pentylfuran, 1-octen-3-ol (mushroom). Grows with temperature; over-long
  cooking degrades it.
- **Wet methods** (boil/steam, ≤100 °C): mostly attenuate — little Maillard,
  volatiles leach into the water.
- Terpenes and other top notes decay under all heat.

This v0 operator is intentionally legible and is the first thing to replace with
a learned model once you have paired raw/cooked GC-MS data.
`LearnedCookingTransform` implements the same interface and drops in behind it.

## Components

```
flavorpair/
  data/     schema (dataclasses), DuckDB store, bulk ingest adapters,
            + resolve (PubChem, deterministic) / extract (LLM, literature)
            / staging (write barrier) / validate (physical checks) / fixtures
  engine/   vectorize (OAV/binary) -> pairing (harmony / novelty)
            + cooking transforms.  PURE MATH, no LLM, no network.
  models/   router (local vs API), local LLM (Ollama/llama.cpp),
            API LLM (DeepSeek/Qwen/Mistral), embeddings + YOUR models
  agent/    tools (deterministic wrappers) + orchestrator (LLM tool-call loop)
  cli.py    pair / partners / profile / coverage / chat
```

Data flow for a chat turn: message → **PARSE** (local LLM → tool calls) →
execute tools (engine) → **EXPLAIN** (router picks local or API → prose).

## Hybrid model layout on an 8GB GPU

| Job | Where | Why |
|---|---|---|
| PARSE (message → query) | local | short, schema-constrained, hot path |
| RESOLVE (fuzzy name → id) | local + embeddings (CPU) | cheap, high volume |
| EXPLAIN (narrate results) | local, → API when large | quality where it shows |
| EXTRACT (GC-MS papers) | **API, batch** | accuracy propagates; off the latency path |
| pairing / cooking | **no model** | deterministic engine |

- **Local:** Qwen2.5-**3B**-Instruct Q4_K_M (~2GB) is the recommended default —
  it parses fine and leaves VRAM for your own embedding/GNN models during
  development. Qwen2.5-**7B**-Instruct Q4_K_M (~4.7GB weights, ~6–7GB with KV
  cache) fits an 8GB card but is tight; use it if you want a stronger parser and
  aren't training alongside it. Run the embedding model (bge-small, ~130MB) on
  CPU either way.
- **API (cheap, for EXPLAIN/overflow and EXTRACT):** DeepSeek-V3
  (`deepseek-chat`) is the low-cost default; Qwen (DashScope) and Mistral
  (`mistral-small`) are drop-in via the same OpenAI-compatible client. Keys from
  env vars.
- The router is the same shape as the coding-agent harness — reuse it.

## Where your own models slot in (`models/embeddings.py`)

1. **Graph-learned ingredient embeddings** — distance reflects shared-compound
   structure, not text. Differentiator over a name lookup.
2. **Link-prediction GNN** over the ingredient–compound bipartite / ingredient
   network — predicts good pairs even for ingredients with sparse compound data.
   The graph is small (~1.5k ingredients, ~37k edges) → trains in minutes on
   your GPU.
3. **Learned CookingTransform** — trained on raw/cooked GC-MS deltas +
   RecipeDB process labels. Same signature as `engine.cooking.apply_cooking`, so
   it drops in behind the interface.
4. **Calibration layer** — proper scoring rules turning raw scores into
   calibrated pairing probabilities (ties to your scoring-rule work). Also the
   right home for extraction confidence, if you attach any: calibrate it against
   validation outcomes rather than reporting a raw model number.

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # 33 tests, no data or network needed

# works immediately against the built-in fixture set:
python -m flavorpair.cli pair strawberry basil
python -m flavorpair.cli partners peach --mode novelty --limit 5
python -m flavorpair.cli profile hazelnut --method roast

# after an ingest:
python -m flavorpair.cli --store data_store/flavorpair.duckdb coverage
python -m flavorpair.cli chat       # needs a local or API model
```

`data/fixtures.py` is ~13 real compounds across 6 ingredients so every command
runs before any ingest. It is **not** reference data — don't tune against it.

## Build order (suggested)

1. `data/ingest.py::ingest_flavordb2` + `store.py` — get real profiles in DuckDB.
   Check `coverage()` afterward; `pct_with_threshold` decides whether OAV
   weighting is usable at all.
2. Validate `engine/pairing.py` against the Barabási cuisine trends — positive
   shared-compound affinity for Western recipes, negative for East Asian. If it
   doesn't reproduce, something is wrong in vectorize or pairing, and you want
   to know that before you have opinions about strawberry and basil.
3. Enrich with FooDB concentrations → turn on OAV weighting. Experimental values
   only; predicted ones will wreck OAV in a way that's hard to notice.
4. Wire the local model for PARSE; the engine already answers headless.
5. Stand up the literature pipeline: `extract.py` → staging → `resolve.py` →
   `promote()`. Work down `rejection_report()` before scaling the crawl.
6. Calibrate `engine/cooking.py` against a few GC-MS papers, then train the
   learned transform on RecipeDB process labels.
7. Add your embedding/GNN re-ranker behind the `PairingModel` interface.

## Status

**Implemented and tested** (33 passing): the whole `engine/` (vectorize,
pairing, cooking), the DuckDB store, the staging tier, the validation gate,
promotion, unit conversion, extraction-response parsing, the agent toolbox, and
the CLI.

**Documented stubs** (`NotImplementedError` with the exact wiring in the
docstring): the five bulk ingest adapters, `extract.extract_document`, the local
LLM clients, and the learned models. Every stub raises rather than silently
falling back — a learned transform that quietly returns the rule-based output is
worse than none, because you can't tell which produced a result.

## Known calibration gaps

- The cooking operators' **relative** magnitudes are unvalidated. On the
  fixtures, roasting hazelnut currently grows the lipid aldehydes slightly
  faster than the pyrazines; real roasted hazelnut is pyrazine-led. The
  aldehyde gain coefficients in `cooking.py` are the knob. This is step 6.
- The **novelty axis is compressed** on small data (all candidates land near
  0.4 on the six fixture ingredients). Re-check its spread once real profiles
  are loaded; if it stays flat, the anchor-saturation constant (`3.0` in
  `score_pair`) is the thing to tune.
- Class-median fallback thresholds in `vectorize.py` are order-of-magnitude
  estimates. They only matter for compounds FlavorDB2 has no threshold for —
  watch `pct_with_threshold`.
