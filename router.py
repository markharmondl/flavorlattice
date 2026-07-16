"""Route each LLM call to local or API based on task type and budget.

The pairing math is deterministic and free (engine/*), so the LLM is only ever
doing three jobs, in rising order of difficulty:

  PARSE    NL request -> structured query {ingredients, method, mode}. Short,
           schema-constrained. A local 3B/7B handles this fine and it's the
           hot path, so keep it local to avoid per-token API cost.

  RESOLVE  fuzzy ingredient name -> canonical ingredient id. Mostly embedding
           lookup (models/embeddings.py); LLM only for tie-breaks. Local.

  EXPLAIN  narrate a ranked result set in prose, optionally grounded in RAG
           over aroma literature. Longer output, benefits from a stronger model.
           Route to API when the result set is large or the user asked "why".

This mirrors the local+cheap-API router in the coding-agent project — same
shape, different tools. Swap the backends in config.yaml.

The router never makes the pairing decision; if an LLM is unavailable the CLI
still returns scored pairs, just without prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Task(str, Enum):
    PARSE = "parse"
    RESOLVE = "resolve"
    EXPLAIN = "explain"


class Backend(str, Enum):
    LOCAL = "local"
    API = "api"


@dataclass
class RouteConfig:
    # send EXPLAIN to API when the result set is at least this large
    explain_api_threshold: int = 10
    # if the local model is loaded and GPU has headroom, prefer it
    prefer_local: bool = True
    force_backend: Backend | None = None   # override for testing / offline


def route(task: Task, *, result_size: int = 0, cfg: RouteConfig | None = None) -> Backend:
    cfg = cfg or RouteConfig()
    if cfg.force_backend is not None:
        return cfg.force_backend
    if task in (Task.PARSE, Task.RESOLVE):
        return Backend.LOCAL
    # EXPLAIN
    if result_size >= cfg.explain_api_threshold:
        return Backend.API
    return Backend.LOCAL if cfg.prefer_local else Backend.API