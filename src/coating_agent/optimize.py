"""Paired-noise grid search on copies of a simulated state only."""
import random
import statistics

def optimize(sim, target: float, seed: int = 42, max_pressure: float = 1.3) -> dict:
    if not .5 <= target <= 1.5 or max_pressure <= 0: raise ValueError("invalid simulated target or constraint")
    start = sim.pump
    candidates = []
    for delta in (-.1, -.05, 0., .05, .1):
        pump = round(start + delta, 3)
        if not .8 <= pump <= 1.2: continue
        runs = []
        for trial in range(3):
            clone = sim.clone(); clone.rng = random.Random(seed + trial); clone.pump = pump; clone.advance(30)
            rows = clone.window(clone.clock, 10)
            pressure = sum(x["pressure"] for x in rows if x["pressure"] is not None) / max(1, sum(x["pressure"] is not None for x in rows))
            flow = sum(x["flow"] for x in rows if x["flow"] is not None) / max(1, sum(x["flow"] is not None for x in rows))
            proxy = sum(x["coating_proxy"] for x in rows) / len(rows)
            runs.append({"pressure": pressure, "flow": flow, "proxy": proxy,
                         "objective": abs(proxy-target) + .05*abs(delta),
                         "feasible": pressure <= max_pressure and .7 <= flow <= 1.3})
        candidates.append({"pump": pump, "runs": runs, "feasible": all(x["feasible"] for x in runs),
                           "objective_mean": statistics.mean(x["objective"] for x in runs),
                           "objective_std": statistics.pstdev(x["objective"] for x in runs)})
    feasible = [x for x in candidates if x["feasible"]]
    baseline = next((x for x in candidates if x["pump"] == start), None)
    return {"baseline": baseline, "candidates": candidates,
            "recommendation": min(feasible, key=lambda x: x["objective_mean"]) if feasible else None,
            "target": target, "seed": seed, "limitation": "仅限模拟器，与实际设备参数无关"}
