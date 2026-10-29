"""Prediction of a synthetic event within 30 minutes, never remaining life."""
from __future__ import annotations
import random
import numpy as np

def label(t: int, event_time: int | None, last_time: int, horizon: int = 30) -> int | None:
    if event_time is not None and t >= event_time: return None
    if event_time is not None and 0 < event_time - t <= horizon: return 1
    if t + horizon > last_time: return None
    return 0

def features(window: list[dict]) -> list[float]:
    if len(window) < 30: raise ValueError("need 30 historical observations")
    rows = window[-30:]
    p = np.array([x["pressure"] if x["pressure"] is not None else np.nan for x in rows], dtype=float)
    q = np.array([x["flow"] if x["flow"] is not None else np.nan for x in rows], dtype=float)
    if np.isnan(p).mean() > .2 or np.isnan(q).mean() > .2: raise ValueError("insufficient observations")
    def metrics(v):
        inds = np.arange(len(v)); ok = ~np.isnan(v)
        return [float(v[ok][-1]), float(np.nanmean(v)), float(np.polyfit(inds[ok], v[ok], 1)[0])]
    return metrics(p) + metrics(q) + [float(np.mean([x["pump"] for x in rows])), float(np.isnan(p).mean())]

def split_assets(ids: list[int], seed: int = 42):
    ids = list(ids); random.Random(seed).shuffle(ids)
    n = len(ids); a, b = int(n*.7), int(n*.85)
    return ids[:a], ids[a:b], ids[b:]
