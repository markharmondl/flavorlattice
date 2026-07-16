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

## Components

```
flavorpair/
  data/     schema (dataclasses), FlavorDB2/FooDB/Flavornet ingest, DuckDB store
  engine/   vectorize (OAV/binary) -> pairing (shared / cosine / novelty)
            + cooking transforms.  PURE MATH, no LLM, no network.
  models/   router (local vs API), local LLM (Ollama/llama.cpp),
            API LLM (DeepSeek/Qwen/Mistral), embeddings + YOUR models
  agent/    tools (deterministic wrappers) + orchestrator (LLM tool-call loop)
  cli.py    pair / partners / profile / chat  (chat = the "chat channel")
```

Data flow for a chat turn: message → **PARSE** (local LLM → tool calls) →
execute tools (engine) → **EXPLAIN** (router picks local or API → prose).

## Hybrid model layout on an 8GB GPU

| Job | Where | Why |
|---|---|---|
| PARSE (message → query) | local | short, schema-constrained, hot path |
| RESOLVE (fuzzy name → id) | local + embeddings (CPU) | cheap, high volume |
| EXPLAIN (narrate results) | local, → API when large | quality where it shows |
| pairing / cooking | **no model** | deterministic engine |

- **Local:** Qwen2.5-**3B**-Instruct Q4_K_M (~2GB) is the recommended default —
  it parses fine and leaves VRAM for your own embedding/GNN models during
  development. Qwen2.5-**7B**-Instruct Q4_K_M (~4.7GB weights, ~6–7GB with KV
  cache) fits an 8GB card but is tight; use it if you want a stronger parser and
  aren't training alongside it. Run the embedding model (bge-small, ~130MB) on
  CPU either way.
- **API (cheap, for EXPLAIN/overflow):** DeepSeek-V3 (`deepseek-chat`) is the
  low-cost default; Qwen (DashScope) and Mistral (`mistral-small`) are
  drop-in via the same OpenAI-compatible client. Keys from env vars.
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
   calibrated pairing probabilities (ties to your scoring-rule work).

## Quick start

```bash
pip install -e .                    # base deps only
python -m pytest tests/ -q          # runs the engine smoke test, no data needed

# after wiring an ingest source + a local model:
flavorpair partners cauliflower --mode harmony --method roast
flavorpair pair strawberry basil
flavorpair chat
```

## Build order (suggested)

1. `data/ingest_flavordb.py` + `store.py` — get real profiles in DuckDB.
2. Validate `engine/pairing.py` against the Barabási cuisine trends.
3. Enrich with FooDB concentrations → turn on OAV weighting.
4. Wire the local model for PARSE; the engine already answers headless.
5. Calibrate `engine/cooking.py` against a few GC-MS papers, then train the
   learned transform on RecipeDB process labels.
6. Add your embedding/GNN re-ranker behind the `PairingModel` interface.

Status: engine core is implemented and tested; ingest, store, and LLM clients
are documented stubs (`NotImplementedError` with the exact wiring in the
docstring).