"""Evaluation diagnostics must separate valid handoffs from planning failure."""
import json
from pathlib import Path
import runpy

import pytest


EVALUATOR = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_agent.py"))
diagnostics = EVALUATOR["planning_diagnostics"]


def trace_for(decision, reason):
    return {"trace": [{"node": "plan"}, {"node": "decision", "detail": decision},
                      {"node": "escalate", "source": decision.get("source"), "detail": {"reason": reason}}]}


def test_fixed_decision_code_has_priority_and_is_reported_only_once():
    decision = {"intent": "escalate", "source": "llm_unavailable",
                "error_code": "model_response_too_many_tools", "warning": "PRIVATE_PROVIDER_TEXT"}
    result = diagnostics(trace_for(decision, "LLM不可用或规划被拒绝（OpenAIRateLimitError）；PRIVATE_PROVIDER_TEXT"))
    assert len(result) == 1
    assert result[0]["error_code"] == "model_response_too_many_tools"
    assert result[0]["category"] == "planning_validation"
    assert "PRIVATE_PROVIDER_TEXT" not in json.dumps(result)


@pytest.mark.parametrize("intent", ["notify_operator", "escalate"])
def test_successful_planned_handoff_is_not_a_planning_warning(intent):
    decision = {"intent": intent, "source": "llm", "action": None}
    assert diagnostics(trace_for(decision, "仅通知人员或停止重复失败动作")) == []


def test_legacy_provider_failure_is_preserved_and_sanitized():
    decision = {"intent": "escalate", "source": "llm_unavailable"}
    result = diagnostics(trace_for(decision, "LLM不可用或规划被拒绝（OpenAIRateLimitError）；PRIVATE_PROVIDER_TEXT"))
    assert result[0]["error_code"] == "provider_rate_limited"
    assert result[0]["exception_type"] == "OpenAIRateLimitError"
    assert "PRIVATE_PROVIDER_TEXT" not in json.dumps(result)


def test_unknown_codes_and_exception_names_do_not_escape_to_reports():
    decision = {"intent": "escalate", "source": "llm_unavailable", "error_code": "PRIVATE_PROVIDER_TEXT"}
    result = diagnostics(trace_for(decision, "PRIVATE_PROVIDER_TEXT"))
    assert result[0]["error_code"] == "planning_unexpected_error"
    assert "PRIVATE_PROVIDER_TEXT" not in json.dumps(result)
    legacy = diagnostics(trace_for({"intent": "escalate", "source": "llm_unavailable"},
                                   "LLM不可用或规划被拒绝（PrivateVendorError）；PRIVATE_PROVIDER_TEXT"))
    assert legacy[0]["exception_type"] == "OtherError"
    assert "PrivateVendorError" not in json.dumps(legacy)


def test_saved_artifact_reclassification_excludes_diagnosis_only():
    valid = {"agent_final_status": "ESCALATED",
             "decisions": [{"intent": "notify_operator", "source": "llm", "action": None}],
             "notifications": [{"title": "需要人员继续处理", "message": "按只诊断目标通知人员"}]}
    failure = {"agent_final_status": "ESCALATED",
               "decisions": [{"intent": "request_approval", "source": "llm"},
                             {"intent": "escalate", "source": "llm_unavailable",
                              "error_code": "model_response_too_many_tools"}],
               "notifications": [{"title": "需要人员继续处理", "message": "PRIVATE_PROVIDER_TEXT"}]}
    assert diagnostics(valid) == []
    result = diagnostics(failure)
    assert len(result) == 1 and result[0]["cycle"] == 2
    assert result[0]["error_code"] == "model_response_too_many_tools"
    assert sum(bool(diagnostics(case)) for case in (valid, failure)) == 1
    assert "PRIVATE_PROVIDER_TEXT" not in json.dumps(result)


def test_old_saved_artifact_without_error_codes_still_classifies_provider_errors():
    legacy = {"agent_final_status": "ESCALATED",
              "decisions": [{"intent": "request_approval", "source": "llm"},
                            {"intent": "escalate", "source": "llm_unavailable"}],
              "notifications": [{"title": "需要人员继续处理",
                                 "message": "LLM不可用或规划被拒绝（PlanningError）；PRIVATE_PROVIDER_TEXT"}]}
    result = diagnostics(legacy)
    assert result[0]["cycle"] == 2 and result[0]["error_code"] == "planning_validation_failed"
    assert "PRIVATE_PROVIDER_TEXT" not in json.dumps(result)


def test_offline_fixture_success_and_diagnosis_only_have_no_planning_failures():
    report = EVALUATOR["run"](include_negative_cases=True)
    assert report["summary"]["passed_cases"] == 8
    assert report["summary"]["cases_with_planning_warnings"] == 0
    assert report["summary"]["model_attempts"] == report["summary"]["model_completed_calls"] == 0
    diagnosis_only = next(case for case in report["cases"] if case["case"] == "diagnosis_only")
    assert diagnosis_only["notifications"] and not diagnosis_only["actions"]


def test_case_interval_applies_only_between_cases_without_retries(monkeypatch, capsys):
    pauses = []
    monkeypatch.setattr(EVALUATOR["time"], "sleep", pauses.append)
    report = EVALUATOR["run"](scenarios=("normal", "sensor_bias"),
                              case_interval_seconds=20, progress=True)
    assert pauses == [20]
    assert report["case_interval_seconds"] == 20
    assert report["summary"]["total_cases"] == report["summary"]["passed_cases"] == 2
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == ["case_started", "case_completed"] * 2
    assert [event["case"] for event in events if event["event"] == "case_started"] == ["normal", "sensor_bias"]


@pytest.mark.parametrize("interval", [-1, 61, float("nan"), float("inf"), True])
def test_invalid_case_intervals_are_rejected_before_any_case(interval, monkeypatch):
    def unexpected_case(*args, **kwargs):
        pytest.fail("invalid pacing must not run a case")
    monkeypatch.setitem(EVALUATOR["run"].__globals__, "evaluate_case", unexpected_case)
    with pytest.raises(ValueError, match="finite"):
        EVALUATOR["run"](case_interval_seconds=interval)
