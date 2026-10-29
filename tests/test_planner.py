import copy

import pytest
from langchain_core.messages import AIMessage

from coating_agent.planner import plan
from coating_agent.simulator import Simulator


def simulated(scenario="blockage", minutes=180):
    sim = Simulator(42, scenario)
    sim.advance(minutes)
    return sim


def run(sim, **kwargs):
    return plan(sim.window(sim.clock, 30), "诊断异常并提出恢复方案", inspect=sim.inspection, **kwargs)


@pytest.mark.parametrize("scenario,action,inspection", [
    ("blockage", "clean_blockage", "T_PRESSURE"),
    ("sensor_bias", "calibrate_pressure", "T_PRESSURE"),
    ("pump_loss", "service_pump", "T_PUMP"),
    ("missing", "restore_acquisition", "T_ACQUISITION"),
    ("mixed", "clean_blockage", "T_PRESSURE"),
])
def test_rules_select_supported_actions_without_mutating_simulation(scenario, action, inspection):
    sim = simulated(scenario)
    before = copy.deepcopy(sim.export_state())
    decision = run(sim)
    assert decision["intent"] == "request_approval"
    assert decision["action"] == action
    assert decision["source"] == "rules" and decision["model_calls"] == 0
    assert {"O1", inspection} <= set(decision["evidence_ids"])
    assert any(i.startswith("K") for i in decision["evidence_ids"])
    assert decision["tool_calls"] == len(decision["tool_results"])
    assert sim.export_state() == before
    assert all("scenario" not in e and "z" not in e for e in decision["evidence"])


def test_mixed_anomaly_replans_after_cleaning_from_new_observations():
    sim = simulated("mixed")
    first = run(sim)
    assert first["action"] == "clean_blockage"
    sim.perform_action(first["action"])
    sim.advance(30)
    second = run(sim, history=[{"action": "clean_blockage", "recovered": False}])
    assert second["action"] == "calibrate_pressure"
    assert second["intent"] == "request_approval"


@pytest.mark.parametrize("objective", ["只诊断，不要执行", "仅做诊断", "请只分析供浆异常", "diagnosis only; do not execute"])
def test_read_only_objectives_never_request_actions(objective):
    sim = simulated()
    result = plan(sim.window(sim.clock, 30), objective, inspect=sim.inspection)
    assert result["intent"] == "notify_operator"
    assert result["action"] is None
    assert not any(t["tool"].startswith("inspect_") for t in result["tool_results"])


def test_normal_short_and_previously_failed_cases():
    assert run(simulated("normal"))["intent"] == "continue_monitoring"
    assert run(simulated(minutes=10))["intent"] == "collect_more_data"
    failed = run(simulated(), history=[{"action": "clean_blockage", "recovered": False}])
    assert failed["intent"] == "escalate" and failed["action"] is None


def test_inspection_must_independently_support_an_action():
    sim = simulated("sensor_bias")
    def no_bias(name):
        result = sim.inspection(name)
        result["data"]["deviation"] = 0
        return result
    result = plan(sim.window(sim.clock, 30), "恢复模拟工况", inspect=no_bias)
    assert result["intent"] == "escalate" and result["action"] is None
    assert plan(sim.window(sim.clock, 30), "恢复模拟工况")["intent"] == "escalate"


def tool_message(*calls):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(i), "type": "tool_call"}
                                             for i, (name, args) in enumerate(calls)])


def install_model(monkeypatch, messages):
    import langchain_openai
    captured = {}
    class FakeModel:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.turn = 0
        def bind_tools(self, tools, **kwargs):
            captured["tools"] = [t.name for t in tools]
            captured.setdefault("tool_sets", []).append([t.name for t in tools])
            captured["bind"] = kwargs
            captured.setdefault("tool_choices", []).append(kwargs.get("tool_choice"))
            return self
        def invoke(self, messages_so_far):
            captured["messages"] = messages_so_far
            if self.turn >= len(messages):
                pytest.fail("model exceeded expected call count")
            response = messages[self.turn]
            self.turn += 1
            if isinstance(response, Exception):
                raise response
            return response
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeModel)
    monkeypatch.setenv("OPENAI_API_KEY", "test-local-fake-key")
    monkeypatch.setenv("COATING_LLM_MODEL", "test-tool-model")
    monkeypatch.delenv("COATING_LLM_API", raising=False)
    return captured


def clean_decision(**overrides):
    result = {"intent": "request_approval", "action": "clean_blockage", "reason": "观测与独立参考支持模拟清理候选。",
              "evidence_ids": ["O1", "K02", "T_PRESSURE"]}
    result.update(overrides)
    return result


def evidence_round():
    return tool_message(("get_statistics", {}), ("search_knowledge", {"query": "模拟 供浆 阻力 压力 流量"}),
                        ("inspect_pressure", {}))


def test_llm_selects_read_tools_then_submits_grounded_decision(monkeypatch):
    captured = install_model(monkeypatch, [evidence_round(), tool_message(("submit_decision", clean_decision()))])
    sim = simulated()
    before = sim.export_state()
    decision = run(sim, mode="llm")
    assert decision["source"] == "llm", decision
    assert decision["intent"] == "request_approval" and decision["action"] == "clean_blockage"
    assert decision["model_calls"] == 2 and decision["tool_calls"] == 4
    assert decision["model_attempts"] == 2 and decision["model_completed_calls"] == 2
    assert captured["kwargs"]["use_responses_api"] is True
    assert captured["bind"]["strict"] is True
    assert captured["tool_choices"] == ["required", "required"]
    assert set(captured["tools"]) == {"get_statistics", "search_knowledge", "inspect_pressure", "inspect_pump", "inspect_acquisition", "submit_decision"}
    assert sim.export_state() == before
    assert any(getattr(m, "type", None) == "tool" for m in captured["messages"])


def test_chat_completions_compatibility_is_explicit(monkeypatch):
    captured = install_model(monkeypatch, [evidence_round(), tool_message(("submit_decision", clean_decision()))])
    monkeypatch.setenv("COATING_LLM_API", "chat_completions")
    assert run(simulated(), mode="llm")["source"] == "llm"
    assert captured["kwargs"]["use_responses_api"] is False


@pytest.mark.parametrize("bad_decision", [
    clean_decision(evidence_ids=["O1", "K02", "T_FABRICATED"]),
    clean_decision(evidence_ids=["O1", "K02"]),
    clean_decision(action="delete_database"),
    clean_decision(intent="continue_monitoring", action=None),
    clean_decision(reason=" "),
    clean_decision(intent="finish", action="clean_blockage"),
])
def test_llm_invalid_decisions_escalate_without_rule_fallback(monkeypatch, bad_decision):
    install_model(monkeypatch, [evidence_round(), tool_message(("submit_decision", bad_decision))])
    result = run(simulated(), mode="llm")
    assert result["intent"] == "escalate" and result["action"] is None
    assert result["source"] == "llm_unavailable"
    assert result["model_calls"] == 2


def test_llm_cannot_invoke_a_write_tool(monkeypatch):
    install_model(monkeypatch, [tool_message(("clean_blockage", {}))])
    sim = simulated()
    before = sim.export_state()
    result = run(sim, mode="llm")
    assert result["source"] == "llm_unavailable" and result["action"] is None
    assert sim.export_state() == before


def test_llm_must_actually_read_cited_evidence(monkeypatch):
    install_model(monkeypatch, [tool_message(("submit_decision", clean_decision()))])
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm_unavailable"
    assert result["tool_results"] == []


def test_llm_normal_text_cannot_replace_structured_submission(monkeypatch):
    install_model(monkeypatch, [AIMessage(content="已修好设备。")])
    result = run(simulated(), mode="llm")
    assert result["intent"] == "escalate" and result["model_calls"] == 1


def test_llm_budget_stops_after_four_rounds(monkeypatch):
    install_model(monkeypatch, [tool_message(("get_statistics", {}))] * 4)
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm_unavailable"
    assert result["model_calls"] == 4
    assert result["action"] is None


def test_llm_failure_does_not_expose_provider_error_or_credentials(monkeypatch):
    install_model(monkeypatch, [RuntimeError("provider key test-local-fake-key in URL")])
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm_unavailable"
    assert "test-local-fake-key" not in str(result)
    assert "provider key" not in str(result)
    assert result["model_attempts"] == 1 and result["model_completed_calls"] == 0


def test_missing_llm_configuration_does_not_run_rules(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm_unavailable" and result["model_calls"] == 0
    assert result["action"] is None and result["tool_results"] == []


def test_llm_read_only_objective_overrides_supported_action(monkeypatch):
    install_model(monkeypatch, [evidence_round(), tool_message(("submit_decision", clean_decision()))])
    sim = simulated()
    result = plan(sim.window(sim.clock, 30), "只诊断，不要执行", mode="llm", inspect=sim.inspection)
    assert result["source"] == "llm"
    assert result["intent"] == "notify_operator" and result["action"] is None


def test_llm_does_not_repeat_a_failed_action(monkeypatch):
    install_model(monkeypatch, [evidence_round(), tool_message(("submit_decision", clean_decision()))])
    result = run(simulated(), mode="llm", history=[{"action": "clean_blockage", "recovered": False}])
    assert result["source"] == "llm" and result["intent"] == "escalate" and result["action"] is None


def test_llm_accepts_all_five_read_tools_in_one_round(monkeypatch):
    all_reads = tool_message(("get_statistics", {}),
                            ("search_knowledge", {"query": "模拟 供浆 阻力 压力 流量"}),
                            ("inspect_pressure", {}), ("inspect_pump", {}), ("inspect_acquisition", {}))
    install_model(monkeypatch, [all_reads, tool_message(("submit_decision", clean_decision()))])
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm" and result["action"] == "clean_blockage"
    assert result["tool_calls"] == 6
    assert {"T_PRESSURE", "T_PUMP", "T_ACQUISITION"} <= {e["id"] for e in result["evidence"]}


def test_last_model_round_is_forced_to_submit_and_has_remaining_budget(monkeypatch):
    captured = install_model(monkeypatch, [
        tool_message(("get_statistics", {})),
        tool_message(("search_knowledge", {"query": "模拟 供浆 阻力 压力 流量"})),
        tool_message(("inspect_pressure", {})),
        tool_message(("submit_decision", clean_decision())),
    ])
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm" and result["model_calls"] == 4
    assert captured["tool_choices"] == ["required", "required", "required", "submit_decision"]
    assert any("这是最后一轮" in str(m.content) for m in captured["messages"])


def test_empty_search_guidance_is_not_treated_as_accessed_evidence(monkeypatch):
    install_model(monkeypatch, [
        tool_message(("get_statistics", {}), ("search_knowledge", {"query": "clean_blockage"}), ("inspect_pressure", {})),
        tool_message(("submit_decision", clean_decision())),
    ])
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm_unavailable" and result["action"] is None
    search_result = next(t["result"] for t in result["tool_results"] if t["tool"] == "search_knowledge")
    assert search_result["cards"] == [] and "中文" in search_result["hint"]
    assert "K02" not in {e["id"] for e in result["evidence"]}


def test_named_submit_choice_uses_langchain_supported_function_schema():
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from coating_agent.planner import DecisionArguments
    submit = StructuredTool.from_function(func=lambda **kwargs: None, name="submit_decision",
                                         description="Submit a plan", args_schema=DecisionArguments)
    model = ChatOpenAI(model="test-model", api_key="test-local-not-a-real-key", use_responses_api=True)
    binding = model.bind_tools([submit], strict=True, tool_choice="submit_decision")
    assert binding.kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit_decision"}}


def test_model_start_callback_can_stop_before_a_network_request(monkeypatch):
    install_model(monkeypatch, [])
    events = []
    def stop(event):
        events.append(event)
        if event["type"] == "model_start":
            raise RuntimeError("cancelled")
    result = run(simulated(), mode="llm", on_event=stop)
    assert result["source"] == "llm_unavailable"
    assert result["model_attempts"] == result["model_completed_calls"] == 0
    assert events == [{"type": "model_start", "attempt": 1, "api": "responses"}]


def test_completed_response_is_counted_before_cancelled_result_is_discarded(monkeypatch):
    install_model(monkeypatch, [evidence_round()])
    events = []
    def stop_after_response(event):
        events.append(event)
        if event["type"] == "model_completed":
            raise RuntimeError("cancelled while request was in flight")
    result = run(simulated(), mode="llm", on_event=stop_after_response)
    assert result["model_attempts"] == result["model_completed_calls"] == 1
    assert result["tool_calls"] == 0
    assert [e["type"] for e in events] == ["model_start", "model_completed"]
    assert not any("response" in e or "content" in e for e in events)


def test_tool_start_callback_stops_before_statistics_are_read(monkeypatch):
    import coating_agent.planner as module
    install_model(monkeypatch, [tool_message(("get_statistics", {}))])
    def unexpected_statistics(window):
        pytest.fail("statistics tool executed after cancellation")
    monkeypatch.setattr(module, "detect", unexpected_statistics)
    events = []
    def stop_before_tool(event):
        events.append(event)
        if event["type"] == "tool_start":
            raise RuntimeError("cancelled")
    result = run(simulated(), mode="llm", on_event=stop_before_tool)
    assert result["tool_calls"] == 0 and result["tool_results"] == []
    assert events[-1] == {"type": "tool_start", "tool": "get_statistics"}


def test_tool_result_callback_preserves_completed_read_and_stops_remaining_batch(monkeypatch):
    install_model(monkeypatch, [evidence_round()])
    events = []
    def stop_after_read(event):
        events.append(event)
        if event["type"] == "tool_result":
            raise RuntimeError("cancelled")
    result = run(simulated(), mode="llm", on_event=stop_after_read)
    assert result["model_attempts"] == result["model_completed_calls"] == 1
    assert result["tool_calls"] == 1
    assert [t["tool"] for t in result["tool_results"]] == ["get_statistics"]
    assert events[-1]["type"] == "tool_result" and events[-1]["result"]["id"] == "O1"
    assert not any(e.get("tool") == "search_knowledge" for e in events)


def test_cancellation_between_rounds_does_not_start_another_request(monkeypatch):
    install_model(monkeypatch, [evidence_round()])
    events = []
    def stop_next_request(event):
        events.append(event)
        if event["type"] == "model_start" and event["attempt"] == 2:
            raise RuntimeError("cancelled")
    result = run(simulated(), mode="llm", on_event=stop_next_request)
    assert result["model_attempts"] == result["model_completed_calls"] == 1
    assert result["tool_calls"] == 3
    assert len([e for e in events if e["type"] == "tool_result"]) == 3


def test_live_events_include_submission_without_model_text_and_do_not_alias_results(monkeypatch):
    install_model(monkeypatch, [evidence_round(), tool_message(("submit_decision", clean_decision()))])
    events = []
    result = run(simulated(), mode="llm", on_event=events.append)
    assert result["source"] == "llm"
    assert len([e for e in events if e["type"] == "model_start"]) == result["model_attempts"] == 2
    assert len([e for e in events if e["type"] == "model_completed"]) == result["model_completed_calls"] == 2
    assert len([e for e in events if e["type"] == "tool_start"]) == result["tool_calls"] == 4
    assert events[-1] == {"type": "tool_result", "tool": "submit_decision", "result": {"accepted": True}}
    observed = next(e for e in events if e["type"] == "tool_result" and e["tool"] == "get_statistics")
    observed["result"]["data"]["pressure_mean"] = 99
    assert result["evidence"][0]["data"]["pressure_mean"] != 99
    assert not any("reasoning" in e or "messages" in e or "content" in e for e in events)


def test_history_does_not_allow_submitting_before_current_cycle_evidence(monkeypatch):
    captured = install_model(monkeypatch, [evidence_round(), tool_message(("submit_decision", clean_decision()))])
    result = run(simulated(), mode="llm", history=[{"action": "clean_blockage", "recovered": False}])
    assert "submit_decision" not in captured["tool_sets"][0]
    assert "submit_decision" in captured["tool_sets"][1]
    assert result["source"] == "llm" and result["intent"] == "escalate"
    assert any("每次规划都是新的证据周期" in str(m.content) for m in captured["messages"])


def test_unseen_current_cycle_submission_has_specific_failure_code(monkeypatch):
    install_model(monkeypatch, [tool_message(("submit_decision", clean_decision()))])
    result = run(simulated(), mode="llm", history=[{"action": "clean_blockage", "recovered": False}])
    assert result["error_code"] == "current_cycle_evidence_missing"
    assert result["tool_calls"] == 0 and result["evidence"] == []


def test_submission_in_same_batch_as_reads_is_rejected_with_specific_code(monkeypatch):
    install_model(monkeypatch, [tool_message(("get_statistics", {}), ("submit_decision", clean_decision()))])
    result = run(simulated(), mode="llm")
    assert result["error_code"] == "decision_mixed_with_reads"
    assert result["tool_calls"] == 0 and result["tool_results"] == []
    assert result["model_attempts"] == result["model_completed_calls"] == 1


@pytest.mark.parametrize("response,code", [
    (AIMessage(content="普通回复不能替代工具调用"), "model_response_no_tools"),
    (AIMessage(content="", invalid_tool_calls=[{"name": "submit_decision", "args": "{bad", "id": "bad", "error": "sensitive raw parse failure", "type": "invalid_tool_call"}]), "model_response_invalid_tools"),
    (tool_message(*[("get_statistics", {})] * 12), "model_response_too_many_tools"),
])
def test_response_protocol_failures_have_distinct_safe_codes(monkeypatch, response, code):
    install_model(monkeypatch, [response])
    result = run(simulated(), mode="llm")
    assert result["error_code"] == code
    assert result["source"] == "llm_unavailable" and result["action"] is None
    assert "sensitive raw parse failure" not in str(result)


def test_rate_limit_is_reported_without_guessing_cause_or_retrying(monkeypatch):
    class OpenAIRateLimitError(RuntimeError):
        pass
    install_model(monkeypatch, [OpenAIRateLimitError("Authorization private request body")])
    result = run(simulated(), mode="llm")
    assert result["error_code"] == "provider_rate_limited"
    assert result["model_attempts"] == 1 and result["model_completed_calls"] == 0
    assert "无法区分原因" in result["warning"] and "未自动重试" in result["warning"]
    assert "Authorization" not in str(result)
    assert "private request body" not in str(result)


def test_callback_interruption_has_a_safe_cancellation_code(monkeypatch):
    class AgentInterrupted(RuntimeError):
        pass
    install_model(monkeypatch, [])
    def stop(event):
        raise AgentInterrupted("private runtime context")
    result = run(simulated(), mode="llm", on_event=stop)
    assert result["error_code"] == "planning_cancelled"
    assert result["model_attempts"] == 0
    assert "private runtime context" not in str(result)


@pytest.mark.parametrize("read_count", [7, 11])
def test_large_legal_read_batch_can_submit_within_total_budget(monkeypatch, read_count):
    reads = [("get_statistics", {}), ("search_knowledge", {"query": "模拟 供浆 阻力 压力 流量"}),
             ("inspect_pressure", {})]
    reads += [("search_knowledge", {"query": "压力 传感器 偏差 校准"})] * (read_count - len(reads))
    captured = install_model(monkeypatch, [tool_message(*reads), tool_message(("submit_decision", clean_decision()))])
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm" and result["action"] == "clean_blockage"
    assert result["tool_calls"] == read_count + 1 <= 12
    assert result["model_attempts"] == result["model_completed_calls"] == 2
    if read_count == 11:
        assert captured["tool_choices"][-1] == "submit_decision"


def test_later_read_batch_is_checked_against_remaining_budget(monkeypatch):
    reads = [("get_statistics", {}), ("search_knowledge", {"query": "模拟 供浆 阻力 压力 流量"}),
             ("inspect_pressure", {})] + [("get_statistics", {})] * 4
    install_model(monkeypatch, [tool_message(*reads), tool_message(*[("get_statistics", {})] * 5)])
    result = run(simulated(), mode="llm")
    assert result["source"] == "llm_unavailable" and result["action"] is None
    assert result["error_code"] == "model_response_too_many_tools"
    assert result["tool_calls"] == 7
    assert result["model_attempts"] == 2
