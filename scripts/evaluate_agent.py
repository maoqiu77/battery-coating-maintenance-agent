"""Reproduce bounded simulated Agent runs without touching the user audit DB.

Approval here is an explicit test fixture, never evidence of user approval or
permission to execute an industrial action. The default planner is offline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from coating_agent.planner import PLANNING_FAILURES
from coating_agent.service import Service


SCENARIOS = ("normal", "blockage", "sensor_bias", "pump_loss", "missing", "mixed")
EXPECTED_ACTIONS = {
    "normal": [], "blockage": ["clean_blockage"],
    "sensor_bias": ["calibrate_pressure"], "pump_loss": ["service_pump"],
    "missing": ["restore_acquisition"], "mixed": ["clean_blockage", "calibrate_pressure"],
}
OBJECTIVE = "检查模拟供浆状态，取得证据并在审批后恢复异常工况；未恢复时重新取证，不能重复执行已失败动作。"
READ_ONLY_OBJECTIVE = "只诊断当前模拟供浆状态并说明证据，不要执行任何处置，也不要申请动作审批。"
MAX_ACTIONS = 4


def planning_diagnostics(agent: dict) -> list[dict]:
    """Classify actual planning errors; ordinary operator handoffs are not errors.

    Current decisions carry fixed error codes. Legacy runtime traces and saved
    evaluation cases are also accepted, without ever copying provider text.
    """
    known_codes = {code: ("planning_validation", message)
                   for code, message in PLANNING_FAILURES.values()}
    known_codes.update({
        # Keep the wording valid for historical per-response limits as well as
        # the newer remaining-budget check sharing this fixed code.
        "model_response_too_many_tools": ("planning_validation", "模型返回的工具数量超过当次规划允许的限制。"),
        "provider_rate_limited": ("rate_limit", "模型服务限流或额度限制，本轮未完成规划。"),
        "provider_timeout": ("timeout", "外部请求超时，本轮未完成规划。"),
        "provider_connection_failed": ("connection", "外部服务连接失败。"),
        "provider_access_denied": ("access", "外部服务拒绝认证或访问权限。"),
        "provider_request_rejected": ("request", "外部服务拒绝请求，需检查接口或模型配置。"),
        "planning_validation_failed": ("planning_validation", "规划未通过本地校验。"),
        "schema_validation_failed": ("planning_validation", "规划或工具参数未通过结构校验。"),
        "planning_cancelled": ("cancelled", "规划已停止，不再发起后续请求。"),
        "planning_unexpected_error": ("provider_or_runtime", "规划或工具异常，原始错误正文未收录。"),
    })
    legacy_types = {
        "PlanningError": "planning_validation_failed", "ValidationError": "schema_validation_failed",
        "RateLimitError": "provider_rate_limited", "OpenAIRateLimitError": "provider_rate_limited",
        "APITimeoutError": "provider_timeout", "OpenAITimeoutError": "provider_timeout",
        "TimeoutError": "provider_timeout", "APIConnectionError": "provider_connection_failed",
        "OpenAIConnectionError": "provider_connection_failed",
        "AuthenticationError": "provider_access_denied", "PermissionDeniedError": "provider_access_denied",
    }

    def diagnostic(code, source, cycle, exception_type=None):
        safe_code = code if isinstance(code, str) and code in known_codes else "planning_unexpected_error"
        category, message = known_codes[safe_code]
        safe_source = source if source in {"rules", "llm", "llm_unavailable", "runtime"} else "unknown"
        return {"cycle": cycle, "source": safe_source, "category": category,
                "error_code": safe_code, "exception_type": exception_type, "message": message}

    trace = agent.get("trace", [])
    if not trace:
        # Evaluation v1/v2 artifacts preserve decisions and notifications rather
        # than the full runtime trace. Reconstruct only their planning boundaries.
        decisions = agent.get("decisions", [])
        if not decisions and isinstance(agent.get("decision"), dict):
            decisions = [agent["decision"]]
        trace = []
        for decision in decisions:
            trace.extend([{"node": "plan"}, {"node": "decision", "detail": decision}])
        if decisions and agent.get("agent_final_status", agent.get("status")) == "ESCALATED":
            notices = [item for item in agent.get("notifications", [])
                       if item.get("title") == "需要人员继续处理"]
            reason = notices[-1].get("message", "") if notices else ""
            trace.append({"node": "escalate", "source": decisions[-1].get("source"),
                          "detail": {"reason": reason}})
    result = []
    cycle = 0
    decision = {}
    reported = False
    for item in trace:
        if item.get("node") == "plan":
            cycle += 1
            decision, reported = {}, False
        if item.get("node") == "decision":
            detail = item.get("detail")
            decision = detail if isinstance(detail, dict) else {}
            if decision.get("error_code"):
                result.append(diagnostic(decision["error_code"], decision.get("source"), cycle))
                reported = True
        if item.get("node") != "escalate":
            continue
        if reported:
            continue
        detail = item.get("detail", {})
        reason = detail.get("reason", "") if isinstance(detail, dict) else ""
        source = decision.get("source", item.get("source"))
        # Compatibility with generated wrappers predating decision.error_code.
        code_match = re.match(r"^(?:LLM规划停止|只读检查停止)（([a-z_]+)）", str(reason))
        if code_match:
            result.append(diagnostic(code_match.group(1), source, cycle))
            reported = True
            continue
        match = re.match(r"^(?:LLM不可用或规划被拒绝|只读检查未完成|运行工具未完成)（([A-Za-z_][A-Za-z0-9_]*)）", str(reason))
        if match:
            name = match.group(1)
            result.append(diagnostic(legacy_types.get(name), source, cycle,
                                     name if name in legacy_types else "OtherError"))
            reported = True
        elif source == "llm_unavailable":
            result.append(diagnostic("planning_unexpected_error", source, cycle))
            reported = True
        elif decision.get("intent") in {"notify_operator", "escalate"} and source in {"rules", "llm"}:
            # A requested diagnosis-only notification or deliberate escalation
            # after a failed action is a valid decision, preserved in decisions
            # and notifications. It is not a failed planning invocation.
            continue
        else:
            result.append({"cycle": cycle, "source": source if source in {"rules", "llm", "runtime"} else "unknown",
                           "category": "workflow_guard", "error_code": None, "exception_type": None,
                           "message": "流程门禁或运行预算结束本次任务，请查看对应轨迹。"})
            reported = True
    return result


def evaluate_case(scenario: str, mode: str, seed: int, *, failure: bool = False,
                  read_only: bool = False) -> dict:
    """Run one isolated case and preserve actual evidence, source and effects."""
    service = Service(":memory:")
    label = "failed_action" if failure else "diagnosis_only" if read_only else scenario
    approvals = []
    error = None
    rid = None
    try:
        rid = service.create(seed, scenario)
        status = service.agent_start(rid, mode=mode,
                                     objective=READ_ONLY_OBJECTIVE if read_only else OBJECTIVE,
                                     max_cycles=4, observation_steps=180)
        while status["agent"]["status"] == "WAITING_APPROVAL":
            if read_only:
                # A read-only objective must never be rescued by fixture approval.
                error = "只诊断目标意外申请了动作审批；未批准任何动作。"
                service.agent_stop(rid)
                break
            if len(approvals) >= MAX_ACTIONS:
                error = "超过评测夹具的四次动作上限，已停止该任务。"
                service.agent_stop(rid)
                break
            pending = status["agent"]["pending_approval"]
            approvals.append({"approval_mode": "test_harness", "action": pending["action"],
                              "approved": True, "simulated_success": not failure,
                              "clock": status["clock"]})
            # Only this isolated in-memory fixture supplies approval automatically.
            # The production UI and runtime still require one explicit approval.
            status = service.agent_resume(rid, pending["id"], approve=True, success=not failure)
    except Exception as exc:
        # Provider errors can contain credentials; record only the exception type.
        error = f"评测调用异常（{type(exc).__name__}），未记录原始外部错误内容。"
    finally:
        report = service.report(rid) if rid is not None else None
        service.db.close()

    if report is None:
        return {"case": label, "scenario": scenario, "seed": seed, "requested_mode": mode,
                "approval_mode": "test_harness", "passed": False, "error": error}
    status = report["status"]
    agent = report["agent"]
    trace = agent.get("trace", [])
    effects = agent.get("effects", [])
    decisions = [item["detail"] for item in trace if item["node"] == "decision"]
    sources = list(dict.fromkeys(item.get("source") for item in decisions))
    actions = [item["action"] for item in effects]
    observations = [item["detail"] for item in trace if item["node"] == "detect"]
    warnings = planning_diagnostics(agent)
    attempts = agent.get("model_attempts", agent.get("model_calls", 0))
    completed = agent.get("model_completed_calls", 0 if attempts == 0 else None)
    source_matches_request = bool(sources) and all(source == mode for source in sources)
    approval_requests = sum(item["node"] == "approval" and item.get("source") == "approval_gate"
                            for item in trace)
    initial = effects[0]["before"] if effects else observations[0] if observations else None
    before = effects[-1]["before"] if effects else initial
    after = effects[-1]["after"] if effects else status["detection"]
    expected_status = "ESCALATED" if failure else "MONITORING" if scenario == "normal" else "RESOLVED"
    if read_only:
        behavioral_pass = not actions and not approvals and not approval_requests and bool(decisions)
        expected = "生成诊断/人员待办，但不申请或执行模拟动作"
    elif failure:
        behavioral_pass = (agent["status"] == "ESCALATED" and actions == ["clean_blockage"]
                           and not effects[0]["recovered"]
                           and any(item["node"] == "replan" for item in trace))
        expected = "执行一次失败清理、复核未恢复、重新规划后转人工；不重复清理"
    else:
        behavioral_pass = (agent["status"] == expected_status
                           and Counter(actions) == Counter(EXPECTED_ACTIONS[scenario]))
        expected = {"agent_status": expected_status, "actions": EXPECTED_ACTIONS[scenario]}
        if mode == "rules":
            behavioral_pass = behavioral_pass and actions == EXPECTED_ACTIONS[scenario]
    return {
        "case": label, "scenario": scenario, "seed": seed, "requested_mode": mode,
        "approval_mode": "test_harness", "automatic_fixture_approval": bool(approvals),
        "expected": expected, "passed": behavioral_pass and source_matches_request and error is None,
        "error": error, "actual_planner_sources": sources,
        "last_planner_source": agent.get("planner_source"),
        "model_calls": agent.get("model_calls", 0), "model_attempts": attempts,
        "model_completed_calls": completed,
        "model_uncompleted_attempts": max(0, attempts - completed) if completed is not None else None,
        "planning_warnings": warnings, "tool_calls": agent.get("tool_calls", 0),
        "planning_cycles": agent.get("cycle", 0), "actions": actions,
        "approval_requests": approval_requests, "approvals": approvals,
        "agent_final_status": agent["status"], "session_final_state": status["state"],
        "clock": status["clock"], "initial_observation": initial, "before": before, "after": after,
        "control": effects[-1]["control"] if effects else None,
        "comparison_scope": "last_action_only" if effects else "no_action_no_counterfactual",
        "effects": effects, "decisions": decisions,
        "inspection_evidence": [item["detail"] for item in trace
                                if item["node"] == "inspect" or
                                (item["node"] == "tool" and item.get("detail", {}).get("tool")
                                 in {"inspect_pressure", "inspect_pump", "inspect_acquisition"})],
        "trace_nodes": [item["node"] for item in trace],
        "notifications": agent.get("notifications", []), "notification_delivery": "local_in_app_only",
        "limitation": "固定种子的简化模拟案例；审批由隔离评测夹具提供，不代表真实人工授权或产线效果。",
    }


def run(mode: str = "rules", seed: int = 42, scenarios: tuple[str, ...] = SCENARIOS,
        include_negative_cases: bool = False, case_interval_seconds: float = 0,
        progress: bool = False) -> dict:
    if (not isinstance(case_interval_seconds, (int, float)) or isinstance(case_interval_seconds, bool)
            or not math.isfinite(case_interval_seconds) or not 0 <= case_interval_seconds <= 60):
        raise ValueError("case_interval_seconds must be finite and between 0 and 60")
    specifications = [(scenario, {}) for scenario in scenarios]
    if include_negative_cases:
        specifications += [("blockage", {"failure": True}), ("sensor_bias", {"read_only": True})]
    cases = []
    for index, (scenario, options) in enumerate(specifications):
        if index and case_interval_seconds:
            time.sleep(case_interval_seconds)
        label = "failed_action" if options.get("failure") else "diagnosis_only" if options.get("read_only") else scenario
        if progress:
            print(json.dumps({"event": "case_started", "index": index + 1, "total": len(specifications),
                              "case": label, "mode": mode}, ensure_ascii=False), flush=True)
        case = evaluate_case(scenario, mode, seed, **options)
        cases.append(case)
        if progress:
            print(json.dumps({"event": "case_completed", "index": index + 1,
                              **{key: case.get(key) for key in
                                 ("case", "passed", "actual_planner_sources", "actions", "agent_final_status",
                                  "model_attempts", "model_completed_calls", "tool_calls", "planning_warnings", "error")}},
                             ensure_ascii=False), flush=True)
    return {
        "schema_version": 2, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode, "seed": seed, "storage": "isolated_in_memory",
        "case_interval_seconds": case_interval_seconds,
        "approval_mode": "test_harness", "max_actions_per_case": MAX_ACTIONS,
        "observation_minutes": 180, "post_action_minutes": 30, "verification_window_minutes": 20,
        "control_design": "相同处置前模拟状态、共同过程随机扰动；处理组和无动作对照分别推进30分钟，比较末20分钟窗口。恢复采集额外涂布量通道使用独立确定性噪声。",
        "summary": {"total_cases": len(cases), "passed_cases": sum(case["passed"] for case in cases),
                    "model_calls": sum(case.get("model_calls", 0) for case in cases),
                    "model_attempts": sum(case.get("model_attempts", 0) for case in cases),
                    "model_completed_calls": (sum(case["model_completed_calls"] for case in cases)
                                              if all(case.get("model_completed_calls") is not None for case in cases) else None),
                    "cases_with_planning_warnings": sum(bool(case.get("planning_warnings")) for case in cases),
                    "tool_calls": sum(case.get("tool_calls", 0) for case in cases)},
        "cases": cases,
        "counter_semantics": "model_calls/model_attempts是发起请求尝试数；model_completed_calls只计成功返回到计划器的响应，旧记录缺失时为null。它们不等于服务端处理量、有效决策数或计费请求数。planning_warnings仅含本地归类及异常类型，不含密钥、服务商错误正文或原始模型输出。",
        "limitation": "此评测验证模拟流程、来源和审批边界，不是准确率评测、真实设备修复率或工业性能提升证明。LLM模式存在外部调用和不确定性；实际source按每次决策记录，未静默替换为规则。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rules", "llm"), default="rules",
                        help="rules离线；llm会调用已配置的外部服务，逐例记录实际决策来源")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append",
                        help="仅运行指定工况，可重复传入；默认全部六种")
    parser.add_argument("--include-negative-cases", action="store_true",
                        help="额外检查动作失败升级和只诊断不审批两个分支")
    parser.add_argument("--case-interval-seconds", type=float, default=0,
                        help="相邻案例之间等待0–60秒，默认0；不自动重试失败请求")
    parser.add_argument("--output", type=Path, default=Path("artifacts/agent-evaluation.json"))
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed必须为0到2^32-1之间的整数")
    if not math.isfinite(args.case_interval_seconds) or not 0 <= args.case_interval_seconds <= 60:
        parser.error("case-interval-seconds必须为0到60之间的有限数值")
    if args.mode == "llm":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    report = run(args.mode, args.seed, tuple(dict.fromkeys(args.scenario or SCENARIOS)),
                 args.include_negative_cases, args.case_interval_seconds, progress=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode": args.mode,
                      "approval_mode": "test_harness", "case_interval_seconds": args.case_interval_seconds,
                      **report["summary"]}, ensure_ascii=False), flush=True)
    return 0 if all(case["passed"] for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
