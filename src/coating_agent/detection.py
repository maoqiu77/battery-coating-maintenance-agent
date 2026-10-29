"""Observable-only signal summaries with explicitly simulation-only thresholds."""
import math


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _slope(rows: list[tuple[int, dict]], key: str) -> float | None:
    if len(rows) < 2:
        return None
    points = [(row.get("minute", index), row[key]) for index, row in rows]
    if not all(_finite(x) for x, _ in points):
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if not denominator:
        return None
    return round(sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator, 6)


def detect(window: list[dict]) -> dict:
    valid = [(i, row) for i, row in enumerate(window)
             if _finite(row.get("pressure")) and _finite(row.get("flow"))]
    completeness = len(valid) / len(window) if window else 0.0
    common = {"sample_count": len(window), "valid_count": len(valid),
              "completeness": round(completeness, 4),
              "pressure_slope": _slope(valid, "pressure"), "flow_slope": _slope(valid, "flow")}
    if not valid or completeness < .8:
        return {**common, "alert": True, "quality": "insufficient", "severity": "unknown",
                "signal_pattern": "insufficient_data", "pressure_mean": None, "flow_mean": None,
                "reason": "数据缺失或无效，无法判断设备状态" if window else "尚无观测，请先推进模拟时间"}
    pressure = sum(row["pressure"] for _, row in valid) / len(valid)
    flow = sum(row["flow"] for _, row in valid) / len(valid)
    high_pressure, low_flow = pressure > 1.17, flow < .84
    pattern = ("pressure_high_flow_low" if high_pressure and low_flow else
               "pressure_high" if high_pressure else "flow_low" if low_flow else "normal")
    reasons = {"pressure_high_flow_low": "模拟压力偏高且流量偏低",
               "pressure_high": "模拟压力偏高，流量未触发低值阈值",
               "flow_low": "模拟流量偏低，压力未触发高值阈值", "normal": "模拟状态正常"}
    return {**common, "alert": high_pressure or low_flow, "quality": "ok",
            "severity": "warning" if high_pressure or low_flow else "normal", "signal_pattern": pattern,
            "pressure_mean": round(pressure, 4), "flow_mean": round(flow, 4), "reason": reasons[pattern]}
