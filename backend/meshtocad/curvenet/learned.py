"""Decisions learned from true CAD faces (the training loop in .cache/r30, synthetic parts and the
benchmark), stored as small gradient-boosted trees in JSON and evaluated here with plain Python."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent


@lru_cache(maxsize=None)
def _model(name: str):
    p = HERE / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def probability(name: str, x) -> float | None:
    """P(yes) of a learned decision for one feature vector, or None if the model is missing."""
    m = _model(name)
    if m is None:
        return None
    raw = m["base"]
    for t in m["trees"]:
        n = 0
        while not t["leaf"][n]:
            v = x[t["f"][n]]
            go_left = t["ml"][n] if v != v else v <= t["t"][n]
            n = t["l"][n] if go_left else t["r"][n]
        raw += t["v"][n]
    return 1.0 / (1.0 + math.exp(-raw))


def threshold(name: str) -> float:
    m = _model(name)
    return float(m.get("threshold", 0.5)) if m else 1.0
