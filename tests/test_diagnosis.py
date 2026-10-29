from coating_agent.knowledge import search
from coating_agent.diagnosis import diagnose, validate_citations, validate_llm_result
from coating_agent.simulator import Simulator
from coating_agent.detection import detect
import pytest


def test_knowledge_retrieval_has_provenance():
    cards = search("压力 流量 堵塞", 3)
    assert cards and all(c["id"] and c["source"] and c["kind"] for c in cards)
    assert search("电动车胎压怎么处理", 5) == []
    assert search("手机电池压力管理", 5) == []
    assert search("儿童血压压力怎么处理", 5) == []


def test_diagnosis_is_uncertain_and_evidence_is_valid():
    sim = Simulator(42, "blockage"); sim.advance(160)
    report = diagnose(sim.window(160, 30))
    assert report["mode"] == "规则模板"
    assert len(report["candidates"]) >= 2
    assert report["uncertainty"]
    assert validate_citations(report)
    report["candidates"][0]["evidence_ids"] = ["not-real"]
    assert not validate_citations(report)


def test_missing_data_does_not_claim_cause():
    sim = Simulator(42, "missing"); sim.advance(90)
    report = diagnose(sim.window(90, 30))
    assert not report["candidates"]
    assert "不足" in report["uncertainty"]

def test_llm_must_use_actual_evidence_and_statistics():
    result = {"candidates": [{"cause": "x", "evidence_ids": ["O1", "K02"], "missing": "x"}], "uncertainty": "不确定", "next_step": "检查"}
    assert not validate_llm_result(result, {"O1"}, True)
    assert not validate_llm_result(result, {"O1", "K02"}, False)
    assert validate_llm_result(result, {"O1", "K02"}, True)
    result["candidates"][0]["evidence_ids"] = ["O1"]
    assert not validate_llm_result(result, {"O1", "K02"}, True)
    result["candidates"] = []
    assert not validate_llm_result(result, {"O1", "K02"}, True)

def test_llm_mode_calls_tools_before_emitting_citations(monkeypatch):
    import json
    from langchain_core.messages import AIMessage
    import langchain_openai
    class FakeModel:
        def __init__(self, **kwargs): self.calls = 0
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[
                    {"name": "get_statistics", "args": {}, "id": "1", "type": "tool_call"},
                    {"name": "search_knowledge", "args": {"query": "模拟 供浆 阻力"}, "id": "2", "type": "tool_call"}])
            return AIMessage(content=json.dumps({"candidates": [{"cause": "可能阻力上升", "evidence_ids": ["O1", "K02"], "missing": "独立检查"}], "uncertainty": "未确定", "next_step": "检查"}, ensure_ascii=False))
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeModel)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key"); monkeypatch.setenv("COATING_LLM_MODEL", "test-model")
    sim = Simulator(42, "blockage"); sim.advance(160)
    result = diagnose(sim.window(160, 30), "llm")
    assert result["mode"] == "LLM" and result["model_calls"] == 2
    assert {"O1", "K02"} <= {e["id"] for e in result["evidence"]}


@pytest.mark.parametrize("scenario,pattern,primary,evidence", [
    ("blockage", "pressure_high_flow_low", "模拟供浆阻力上升", "K02"),
    ("sensor_bias", "pressure_high", "压力传感器偏差", "K03"),
    ("pump_loss", "flow_low", "模拟泵效率下降", "K08"),
    ("mixed", "pressure_high_flow_low", "模拟供浆阻力上升", "K02"),
])
def test_observed_signal_patterns_prioritize_different_causes(scenario, pattern, primary, evidence):
    sim = Simulator(42, scenario)
    sim.advance(180)
    result = diagnose(sim.window(180, 30))
    assert result["signal_pattern"] == pattern
    assert result["severity"] == "warning"
    assert result["summary"]
    assert result["candidates"][0]["cause"] == primary
    assert evidence in result["candidates"][0]["evidence_ids"]
    assert all(len(c["evidence_ids"]) >= 2 for c in result["candidates"])
    assert validate_citations(result)
    assert all(c["title"] and c["version"] and c["limitation"] for c in result["evidence"][1:])


@pytest.mark.parametrize("scenario,pattern,severity", [
    ("normal", "normal", "normal"), ("missing", "insufficient_data", "unknown")
])
def test_normal_and_missing_windows_do_not_invent_causes_or_call_llm(monkeypatch, scenario, pattern, severity):
    import coating_agent.diagnosis as module
    def unexpected_call(state):
        pytest.fail("An LLM should not be called without a usable anomalous window")
    monkeypatch.setattr(module, "llm_diagnosis", unexpected_call)
    sim = Simulator(42, scenario)
    sim.advance(90)
    result = diagnose(sim.window(90, 30), "llm")
    assert result["signal_pattern"] == pattern
    assert result["severity"] == severity
    assert result["candidates"] == []
    assert result["model_calls"] == 0
    assert "未调用" in result["warning"]
    assert result["next_step"]


def test_detection_uses_elapsed_minutes_and_reports_invalid_samples():
    window = [{"minute": 2, "pressure": 1., "flow": 1.},
              {"minute": 4, "pressure": float("nan"), "flow": 1.},
              {"minute": 6, "pressure": 1.4, "flow": .8},
              {"minute": 8, "pressure": 1.6, "flow": .7},
              {"minute": 10, "pressure": 1.8, "flow": .6}]
    result = detect(window)
    assert result["sample_count"] == 5 and result["valid_count"] == 4
    assert result["completeness"] == .8 and result["quality"] == "ok"
    assert result["pressure_slope"] == pytest.approx(.1)
    assert result["flow_slope"] == pytest.approx(-.05)
    window[0]["flow"] = float("inf")
    result = detect(window)
    assert result["completeness"] == .6 and result["quality"] == "insufficient"
    assert result["pressure_mean"] is None
    assert detect([])["signal_pattern"] == "insufficient_data"


@pytest.mark.parametrize("bad_result", [None, [], "text", {"candidates": [None]},
    {"candidates": [{"cause": " ", "missing": "x", "evidence_ids": ["O1", "K02"]}],
     "uncertainty": "x", "next_step": "x"},
    {"candidates": [{"cause": "x", "missing": "x", "evidence_ids": [["O1"], "K02"]}],
     "uncertainty": "x", "next_step": "x"},
])
def test_malformed_model_results_are_rejected_without_validator_errors(bad_result):
    assert not validate_llm_result(bad_result, {"O1", "K02"}, True)
    assert not validate_citations(bad_result)


def test_invalid_llm_output_falls_back_to_observation_specific_result(monkeypatch):
    import coating_agent.diagnosis as module
    def invalid_result(state):
        raise ValueError("invalid response that must never reach the UI")
    monkeypatch.setattr(module, "llm_diagnosis", invalid_result)
    sim = Simulator(42, "sensor_bias")
    sim.advance(180)
    result = diagnose(sim.window(180, 30), "llm")
    assert result["mode"] == "规则模板"
    assert result["candidates"][0]["cause"] == "压力传感器偏差"
    assert "回退" in result["warning"]
    assert "invalid response" not in result["warning"]
