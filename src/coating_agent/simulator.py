"""A transparent, deliberately simplified and dimensionless simulator."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, asdict


@dataclass
class Observation:
    minute: int
    pump: float
    web_speed: float
    pressure: float | None
    flow: float | None
    temperature: float
    coating_proxy: float | None


class Simulator:
    def __init__(self, seed: int = 42, scenario: str = "blockage"):
        if scenario not in {"normal", "blockage", "sensor_bias", "pump_loss", "missing", "mixed"}:
            raise ValueError("unknown scenario")
        self.seed, self.scenario = seed, scenario
        self.rng = random.Random(seed)
        self.clock = 0
        self.z, self.eta, self.pump, self.web_speed = 0.0, 1.0, 1.0, 1.0
        self.p, self.q = 1.0, 1.0
        self.p_base = self.rng.uniform(.97, 1.03)
        self.q_base = self.rng.uniform(.97, 1.03)
        self.a = self.rng.uniform(.8, 1.0)
        self.b = self.rng.uniform(.8, 1.0)
        self.failure_z = 1.2
        self.event_time = None
        self._cleaned = False
        self._pressure_calibrated = False
        self._pump_serviced = False
        self._acquisition_restored = False
        self._observations: list[dict] = []

    def advance(self, n: int = 1) -> None:
        if not isinstance(n, int) or not 1 <= n <= 3600:
            raise ValueError("invalid number of steps")
        for _ in range(n):
            self.clock += 1
            if self.clock >= 60 and self.scenario in {"blockage", "mixed"}:
                # Draw the same disturbance even after cleaning, so treated and
                # untreated clones keep a common process-noise sequence.
                growth = .009 * self.rng.uniform(.8, 1.2)
                if not self._cleaned:
                    self.z += growth
            if not self._pump_serviced and self.scenario == "pump_loss" and self.clock >= 60:
                self.eta = max(.55, self.eta - .002)
            if self.z >= self.failure_z and self.event_time is None:
                self.event_time = self.clock
            q_target = self.q_base * self.pump * self.eta / (1 + self.a * self.z)
            p_target = self.p_base + self.b * self.pump * self.eta * self.z / (1 + self.a * self.z)
            self.q += .2 * (q_target - self.q) + self.rng.gauss(0, .002)
            self.p += .2 * (p_target - self.p) + self.rng.gauss(0, .002)
            bias = .4 if (not self._pressure_calibrated and self.clock >= 60
                          and self.scenario in {"sensor_bias", "mixed"}) else 0
            pressure = round(self.p + bias + self.rng.gauss(0, .005), 5)
            flow = round(self.q + self.rng.gauss(0, .005), 5)
            originally_missing = self.scenario == "missing" and self.clock % 3 != 0
            if not self._acquisition_restored and originally_missing:
                pressure = flow = None
            if flow is None:
                proxy = None
            else:
                # A newly restored sample has no proxy-noise draw in its
                # untreated counterpart. Use a separate deterministic stream
                # for that extra channel, preserving the process RNG state.
                proxy_rng = (random.Random(f"{self.seed}:restored-proxy:{self.clock}")
                             if originally_missing else self.rng)
                proxy = round(flow / self.web_speed + proxy_rng.gauss(0, .005), 5)
            self._observations.append(asdict(Observation(self.clock, self.pump, self.web_speed,
                pressure, flow, round(1 + .04 * self.z + self.rng.gauss(0, .01), 3), proxy)))

    def window(self, end: int, size: int) -> list[dict]:
        if not isinstance(end, int) or not isinstance(size, int) or end < 1 or end > self.clock or size < 1:
            raise ValueError("invalid or future window")
        return copy.deepcopy(self._observations[max(0, end-size):end])

    def clean(self, success: bool) -> None:
        """Compatibility entry point for the original simulated cleaning action."""
        self.perform_action("clean_blockage", success)

    def perform_action(self, action: str, success: bool = True) -> None:
        """Change only the simulator; callers must enforce approval and audit.

        Past observations and the simulation clock are never rewritten. A failed
        action has no effect, so subsequent independent verification can fail.
        """
        if action not in {"clean_blockage", "calibrate_pressure", "service_pump", "restore_acquisition"}:
            raise ValueError("unknown simulated action")
        if not isinstance(success, bool):
            raise ValueError("simulated action success must be boolean")
        if not success:
            return
        if action == "clean_blockage":
            self.z *= .05
            self._cleaned = True
        elif action == "calibrate_pressure":
            self._pressure_calibrated = True
        elif action == "service_pump":
            self.eta = 1.0
            self._pump_serviced = True
        elif action == "restore_acquisition":
            self._acquisition_restored = True

    def inspection(self, name: str) -> dict:
        """Read a deterministic, explicitly simulated independent check.

        The virtual pressure reference and unloaded pump bench are additional
        simulated measurements, not real instruments. Inspection never consumes
        the process RNG or advances time, so paired counterfactual runs retain
        their existing random stream. No scenario or hidden fault state is
        included in the returned evidence.
        """
        ids = {"inspect_pressure": "T_PRESSURE", "inspect_pump": "T_PUMP",
               "inspect_acquisition": "T_ACQUISITION"}
        if name not in ids:
            raise ValueError("unknown simulated inspection")
        rows = self.window(self.clock, 30) if self.clock else []
        if name == "inspect_pressure":
            measured = rows[-1]["pressure"] if rows else None
            reference = round(self.p, 5) if rows else None
            data = {"measured_pressure": measured, "reference_pressure": reference,
                    "deviation": round(measured - reference, 5) if measured is not None else None}
            limitation = "模拟独立压力参考与同分钟采集读数对照；非真实校准仪器，缺失读数不推断偏差。"
        elif name == "inspect_pump":
            # A virtual unloaded bench isolates pump delivery from line resistance.
            # Emit measured quantities and their ratio, not a hidden fault label.
            expected = self.q_base * self.pump
            measured = expected * self.eta
            data = {"expected_flow": round(expected, 5) if rows else None,
                    "measured_flow": round(measured, 5) if rows else None,
                    "efficiency_ratio": round(measured / expected, 5) if rows and expected > 0 else None}
            limitation = "模拟独立空载台架流量检查，假设排除流路阻力；非真实泵性能测试或维修依据。"
        else:
            def valid(row):
                return all(isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)
                           and math.isfinite(row[key]) for key in ("pressure", "flow"))
            count = sum(valid(row) for row in rows)
            data = {"completeness": round(count / len(rows), 4) if rows else 0.0,
                    "sample_count": len(rows), "valid_count": count}
            limitation = "仅统计最近最多30个模拟观测的有效性；恢复采集仅影响后续数据，不补写历史缺失。"
        return {"id": ids[name], "kind": "模拟检查", "tool": name, "clock": self.clock,
                "data": data, "limitation": limitation}

    def clone(self) -> "Simulator":
        return copy.deepcopy(self)

    def snapshot(self) -> dict:
        return {"clock": self.clock, "observations": self.window(self.clock, 1) if self.clock else []}

    def export_state(self) -> dict:
        state = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "rng"}
        state["random_state"] = self.rng.getstate()
        return state

    @classmethod
    def from_state(cls, state: dict) -> "Simulator":
        sim = cls(state["seed"], state["scenario"])
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        for key, value in state.items():
            if key != "random_state": setattr(sim, key, value)
        sim.rng.setstate(tuples(state["random_state"]))
        return sim
