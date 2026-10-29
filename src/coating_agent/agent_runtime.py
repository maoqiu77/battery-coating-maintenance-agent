"""Bounded, inspectable LangGraph orchestration for simulated maintenance.

The graph owns the sequence of tools, not actuator authority. Every simulated
change still requires a fresh approval bound to one session version.
"""
from __future__ import annotations

import copy
import secrets
import time
from typing import TYPE_CHECKING, TypedDict
import uuid

from langgraph.graph import END, START, StateGraph

from coating_agent.detection import detect
from coating_agent.diagnosis import diagnose

if TYPE_CHECKING:
    from coating_agent.service import Service


ACTION_LABELS = {
    "clean_blockage": "模拟清理供浆阻力",
    "calibrate_pressure": "模拟校准压力传感器",
    "service_pump": "模拟维护供浆泵",
    "restore_acquisition": "模拟恢复采集链路",
}
INSPECTION_NAMES = {"inspect_pressure", "inspect_pump", "inspect_acquisition"}
ACTIVE = {"RUNNING", "WAITING_APPROVAL"}


def default_agent() -> dict:
    return {"status": "IDLE", "mode": "rules", "objective": "", "run_id": None,
            "cycle": 0, "max_cycles": 4, "decision": None, "pending_approval": None,
            "trace": [], "notifications": [], "effects": [], "model_calls": 0,
            "model_attempts": 0, "model_completed_calls": 0, "tool_calls": 0, "planner_source": "rules"}


def approval_expired(agent: dict) -> bool:
    pending = agent.get("pending_approval")
    return bool(agent.get("status") == "WAITING_APPROVAL" and pending
                and (time.time() > pending["expires_at"]
                     or time.monotonic() > agent.get("_approval_deadline", float("inf"))))


def active(agent: dict) -> bool:
    return agent.get("status") in ACTIVE and not approval_expired(agent)


def public_agent(agent: dict) -> dict:
    public = copy.deepcopy({key: value for key, value in agent.items() if not key.startswith("_")})
    if approval_expired(agent):
        public["status"] = "ESCALATED"
        public["pending_approval"] = None
        public["approval_expired"] = True
        # This derived notice is stable and does not turn status polling into writes.
        public["notifications"].append({"id": f"expired-{agent['run_id']}", "level": "warning",
            "title": "审批已过期", "message": "本次操作没有执行；请重新启动 Agent 取证并申请审批。",
            "acknowledged": False, "clock": agent.get("_clock", 0)})
    return public


class AgentInterrupted(ValueError):
    """A stop or another session mutation invalidated an in-flight plan."""


class RuntimeState(TypedDict, total=False):
    runner: "AgentRunner"
    entry: str
    route: str


class AgentRunner:
    def __init__(self, service: "Service", rid: str, run_id: str, version: int,
                 action: str | None = None, success: bool = True):
        self.service, self.rid, self.run_id, self.version = service, rid, run_id, version
        self.action, self.success = action, success
        self.inspections: dict[str, dict] = {}
        self.window: list[dict] = []
        self.statistics: dict = {}

    def _get(self):
        s = self.service._get(self.rid)
        if (s["agent"]["run_id"] != self.run_id or s["agent"]["status"] != "RUNNING"
                or s["version"] != self.version):
            raise AgentInterrupted("Agent 已停止或会话已变化，旧结果已丢弃")
        return s, s["agent"]

    def _save(self, s):
        s["version"] += 1
        self.version = s["version"]
        s["agent"]["_clock"] = s["sim"].clock
        self.service._persist(self.rid)
        self.service.db.commit()

    def _trace(self, s, node: str, label: str, detail, source: str = "runtime"):
        item = {"node": node, "label": label, "detail": detail, "clock": s["sim"].clock,
                "timestamp": time.time(), "source": source, "agent_run_id": self.run_id}
        s["agent"]["trace"].append(item)
        self.service._event(self.rid, "agent_trace", item)

    def _notify(self, s, level: str, title: str, message: str):
        notification = {"id": uuid.uuid4().hex, "level": level, "title": title,
                        "message": message, "acknowledged": False, "clock": s["sim"].clock}
        s["agent"]["notifications"].append(notification)
        self.service._event(self.rid, "agent_notification", notification)

    def observe(self):
        with self.service.lock:
            s, agent = self._get()
            target = agent["_observation_target"]
            remaining = max(0, target - s["sim"].clock)
            while remaining:
                steps = min(30, remaining)
                s["sim"].advance(steps)
                remaining -= steps
                agent["tool_calls"] += 1
                stats = detect(s["sim"].window(s["sim"].clock, 30))
                self._trace(s, "observe", "采集模拟观测", {"steps": steps, "detection": stats}, "simulator")
            self.window = s["sim"].window(s["sim"].clock, 30) if s["sim"].clock else []
            agent["tool_calls"] += 1
            self._trace(s, "observe", "读取当前观测窗口", {"samples": len(self.window)}, "observation_tool")
            self._save(s)
        return "detect"

    def detect(self):
        with self.service.lock:
            s, _ = self._get()
            self.statistics = detect(self.window)
            s["state"] = "ALERTED" if self.statistics["alert"] else "OBSERVING"
            self._trace(s, "detect", "检测异常与数据质量", self.statistics, "detection_rules")
            self._save(s)
        return "gather"

    def gather(self):
        # This baseline evidence report is separate from the selected planner.
        result = diagnose(self.window, "offline")
        with self.service.lock:
            s, agent = self._get()
            s["diagnosis"] = result
            s["diagnosis_clock"] = s["sim"].clock
            s["state"] = "DIAGNOSED"
            agent["tool_calls"] += 1
            self._trace(s, "gather", "检索知识并建立证据链",
                        {"summary": result["summary"], "evidence_ids": [e["id"] for e in result["evidence"]]}, "knowledge_tool")
            self._save(s)
        return "plan"

    def inspect(self, name: str):
        with self.service.lock:
            s, agent = self._get()
            if name not in INSPECTION_NAMES:
                raise ValueError("不支持的只读检查工具")
            if name in self.inspections:
                return copy.deepcopy(self.inspections[name])
            if len(self.inspections) >= 3:
                raise ValueError("本轮只读检查预算已用尽")
            result = s["sim"].inspection(name)
            self.inspections[name] = result
            return copy.deepcopy(result)

    def planner_event(self, event: dict):
        """Record attempts when they occur and cancel before further work.

        An HTTP response already in flight may arrive after stop. Its completion
        remains auditable, but cannot revive the task or update another run.
        """
        with self.service.lock:
            kind = event["type"]
            if kind in {"model_start", "tool_start"}:
                s, agent = self._get()
                if kind == "model_start":
                    agent["model_attempts"] = agent.get("model_attempts", agent["model_calls"]) + 1
                    agent["model_calls"] = agent["model_attempts"]
                    self._trace(s, "model", "发起模型请求", event, "llm")
                else:
                    agent["tool_calls"] += 1
                    self._trace(s, "tool_start", "开始调用只读工具", {"tool": event["tool"]}, agent["mode"])
                self._save(s)
                return

            if kind not in {"model_completed", "tool_result"}:
                raise ValueError("unknown planner event")
            s = self.service._get(self.rid)
            agent = s["agent"]
            same_run = agent["run_id"] == self.run_id
            valid = same_run and agent["status"] == "RUNNING" and s["version"] == self.version
            if same_run:
                if kind == "model_completed":
                    agent["model_completed_calls"] = agent.get("model_completed_calls", 0) + 1
                    self._trace(s, "model", "模型请求返回", event, "llm")
                else:
                    self._trace(s, "tool", "计划器只读工具结果",
                                {"tool": event["tool"], "result": event["result"]}, agent["mode"])
                self._save(s)
            else:
                self.service._event(self.rid, "agent_late_event", {"agent_run_id": self.run_id,
                    "clock": s["sim"].clock, "event": copy.deepcopy(event)})
                self.service.db.commit()
            if not valid:
                raise AgentInterrupted("旧任务已停止；已记录在途请求结果，停止后续规划")

    def _action_allowed(self, action: str):
        stats = self.statistics
        data = {name: item.get("data", {}) for name, item in self.inspections.items()}
        pressure = data.get("inspect_pressure", {})
        pump = data.get("inspect_pump", {})
        acquisition = data.get("inspect_acquisition", {})
        if action == "clean_blockage":
            return (stats.get("quality") == "ok" and (stats.get("pressure_mean") or 0) > 1.17
                    and (stats.get("flow_mean") or 0) < .84 and (pressure.get("reference_pressure") or 0) > 1.17)
        if action == "calibrate_pressure":
            return (stats.get("quality") == "ok" and (stats.get("pressure_mean") or 0) > 1.17
                    and abs(pressure.get("deviation") or 0) > .15)
        if action == "service_pump":
            return (stats.get("quality") == "ok" and (stats.get("flow_mean") or 0) < .84
                    and pump.get("efficiency_ratio") is not None and pump["efficiency_ratio"] < .9)
        if action == "restore_acquisition":
            return stats.get("quality") != "ok" and acquisition.get("completeness", 1) < .8
        return False

    def plan(self):
        from coating_agent.planner import plan
        with self.service.lock:
            s, agent = self._get()
            if agent["cycle"] >= agent["max_cycles"]:
                return self._escalate_locked(s, "已达到规划轮次上限，需要人员继续检查。")
            agent["cycle"] += 1
            self.inspections = {}
            objective, mode, history = agent["objective"], agent["mode"], copy.deepcopy(agent["effects"])
            self._trace(s, "plan", "规划下一步", {"cycle": agent["cycle"], "mode": mode}, mode)
            self._save(s)
        result = plan(self.window, objective, mode=mode, inspect=self.inspect, history=history,
                      on_event=self.planner_event)
        with self.service.lock:
            s, agent = self._get()
            agent["planner_source"] = result.get("source", mode)
            agent["decision"] = {key: copy.deepcopy(result.get(key))
                                 for key in ("intent", "reason", "action", "evidence_ids", "source")}
            # The planner supplies only fixed diagnostic codes and sanitized warnings.
            # Keep these in the audit report as well as the operator notification.
            for key in ("error_code", "warning"):
                if key in result:
                    agent["decision"][key] = copy.deepcopy(result[key])
            agent["evidence"] = copy.deepcopy(result.get("evidence", []))
            self._trace(s, "decision", "形成下一步决策", agent["decision"], agent["planner_source"])
            intent, action = result.get("intent"), result.get("action")
            if intent == "request_approval":
                if action not in ACTION_LABELS or not self._action_allowed(action):
                    return self._escalate_locked(s, "所提议动作未通过独立观测门禁，已阻止执行并转人工检查。")
                self.action = action
                self._save(s)
                return "approval"
            if intent == "collect_more_data":
                if agent["cycle"] >= agent["max_cycles"]:
                    return self._escalate_locked(s, "追加观测后仍无法决策，已达到本次轮次上限。")
                agent["_observation_target"] = s["sim"].clock + 30
                self._trace(s, "replan", "追加独立观测", {"steps": 30}, agent["planner_source"])
                self._save(s)
                return "observe"
            if intent in {"continue_monitoring", "finish"} and not self.statistics["alert"]:
                agent["status"] = "MONITORING"
                s["state"] = "OBSERVING"
                self._notify(s, "info", "本轮观察完成", "当前窗口未见模拟阈值异常；本次有界运行结束，可再次启动以继续观察。")
                self._trace(s, "monitor", "结束本次观察", {"detection": self.statistics}, agent["planner_source"])
                self._save(s)
                return "end"
            return self._escalate_locked(s, result.get("warning") or result.get("reason") or "当前证据不足以安全选择下一步。")

    def approval(self):
        with self.service.lock:
            s, agent = self._get()
            agent["status"] = "WAITING_APPROVAL"
            agent["pending_approval"] = {"id": secrets.token_urlsafe(24), "action": self.action,
                                         "label": ACTION_LABELS[self.action], "expires_at": time.time() + 120,
                                         "version": s["version"] + 1}
            agent["_approval_deadline"] = time.monotonic() + 120
            s["state"] = "AWAITING_CONFIRMATION"
            s["token"] = None
            self._trace(s, "approval", "等待人工审批", {"action": self.action, "expires_seconds": 120}, "approval_gate")
            self._notify(s, "warning", "需要审批模拟动作", f"{ACTION_LABELS[self.action]}：确认后才会执行，并自动进行独立窗口复核。")
            self._save(s)
        return "end"

    def execute(self):
        with self.service.lock:
            s, agent = self._get()
            before = detect(s["sim"].window(s["sim"].clock, 30))
            control = s["sim"].clone()
            control.advance(30)
            control_stats = detect(control.window(control.clock, 20))
            s["sim"].perform_action(self.action, self.success)
            s["has_execution"] = True
            s["state"] = "ACTION_EXECUTED"
            s["diagnosis_clock"] = None
            agent["tool_calls"] += 1
            agent["_execution"] = {"action": self.action, "label": ACTION_LABELS[self.action],
                                   "before": before, "control": control_stats, "before_clock": s["sim"].clock}
            self._trace(s, "execute", "执行已批准的模拟动作", {"action": self.action, "label": ACTION_LABELS[self.action]}, "simulation_action")
            self.service._event(self.rid, "action_executed", {"action": self.action, "agent_run_id": self.run_id,
                                                             "clock": s["sim"].clock, "simulated_success": self.success})
            self._save(s)
        return "verify"

    def verify(self):
        with self.service.lock:
            s, agent = self._get()
            s["sim"].advance(30)
            after = detect(s["sim"].window(s["sim"].clock, 20))
            recovered = after["quality"] == "ok" and not after["alert"]
            effect = {**agent.pop("_execution"), "after": after, "recovered": recovered, "clock": s["sim"].clock}
            agent["effects"].append(effect)
            agent["tool_calls"] += 1
            s["state"] = "RESOLVED" if recovered else "UNRESOLVED"
            self._trace(s, "verify", "独立观测窗口复核", effect, "verification_tool")
            self.service._event(self.rid, "verification", {"state": s["state"], "before_clock": effect["before_clock"],
                                                          "after_clock": s["sim"].clock, "detection": after})
            if recovered:
                agent["status"] = "RESOLVED"
                self._notify(s, "success", "模拟工况已恢复", "处置后独立 20 分钟窗口通过复核；报告已保存动作、对照和证据。")
                self._trace(s, "resolved", "完成本次运维闭环", {"cycles": agent["cycle"]}, "verification_tool")
                self._save(s)
                return "end"
            if agent["cycle"] >= agent["max_cycles"]:
                return self._escalate_locked(s, "独立复核仍有异常，且已达到规划轮次上限，请人员继续检查。")
            self._trace(s, "replan", "复核未恢复，重新取证规划", {"remaining_cycles": agent["max_cycles"] - agent["cycle"]}, "runtime")
            agent["_observation_target"] = s["sim"].clock
            self._save(s)
        return "observe"

    def _escalate_locked(self, s, reason: str):
        agent = s["agent"]
        agent["status"] = "ESCALATED"
        agent["pending_approval"] = None
        if s["state"] != "ACTION_EXECUTED":
            s["state"] = "UNRESOLVED"
        self._trace(s, "escalate", "转人工处理", {"reason": reason}, agent["planner_source"])
        self._notify(s, "warning", "需要人员继续处理", reason)
        self._save(s)
        return "end"

    def run(self, entry: str):
        try:
            workflow.invoke({"runner": self, "entry": entry}, {"recursion_limit": 80})
        except AgentInterrupted:
            pass
        except Exception as exc:
            with self.service.lock:
                try:
                    s, _ = self._get()
                except AgentInterrupted:
                    pass
                else:
                    self._escalate_locked(s, f"运行工具未完成（{type(exc).__name__}），没有自动重试变更，请人工检查。")
        return self.service.status(self.rid)


def _node(name: str):
    def step(state: RuntimeState):
        return {"route": getattr(state["runner"], name)()}
    return step


graph = StateGraph(RuntimeState)
for name in ("observe", "detect", "gather", "plan", "approval", "execute", "verify"):
    graph.add_node(name, _node(name))
graph.add_conditional_edges(START, lambda state: state["entry"], {"observe": "observe", "execute": "execute"})
graph.add_edge("observe", "detect")
graph.add_edge("detect", "gather")
graph.add_edge("gather", "plan")
graph.add_conditional_edges("plan", lambda state: state["route"], {"observe": "observe", "approval": "approval", "end": END})
graph.add_edge("approval", END)
graph.add_edge("execute", "verify")
graph.add_conditional_edges("verify", lambda state: state["route"], {"observe": "observe", "end": END})
workflow = graph.compile()
