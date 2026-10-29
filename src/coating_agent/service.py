"""Local simulation sessions, deterministic action gate and SQLite audit."""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
import uuid
from functools import wraps
from pathlib import Path
from threading import RLock

from coating_agent.simulator import Simulator
from coating_agent.detection import detect
from coating_agent.diagnosis import diagnose
from coating_agent.agent_runtime import AgentRunner, active, approval_expired, default_agent, public_agent

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "xgboost.joblib"


class SessionNotFound(ValueError):
    pass


def locked(fn):
    """Serialize access; reads deliberately do not rewrite sessions."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            try:
                return fn(self, *args, **kwargs)
            finally:
                # Denied actions are audited too. Reads start no transaction.
                if self.db.in_transaction:
                    self.db.commit()
    return wrapper


class Service:
    def __init__(self, db: str = "audit.sqlite3"):
        self.sessions = {}
        self.lock = RLock()
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS audit (run_id TEXT, event TEXT, detail TEXT, ts REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS audit_run_id ON audit(run_id)")
        self.db.commit()
        for rid, payload in self.db.execute("SELECT run_id, payload FROM sessions").fetchall():
            saved = json.loads(payload)
            sim = Simulator.from_state(saved["sim"])
            executed = self.db.execute(
                "SELECT 1 FROM audit WHERE run_id=? AND event='action_executed' LIMIT 1", (rid,)
            ).fetchone() is not None
            created = self.db.execute("SELECT MIN(ts) FROM audit WHERE run_id=?", (rid,)).fetchone()[0]
            diagnosis_clock = saved.get("diagnosis_clock")
            if "diagnosis_clock" not in saved and saved["diagnosis"]:
                row = self.db.execute(
                    "SELECT detail FROM audit WHERE run_id=? AND event='diagnosis' ORDER BY rowid DESC LIMIT 1", (rid,)
                ).fetchone()
                diagnosis_clock = json.loads(row[0]).get("clock") if row else None
            state = saved["state"]
            agent = saved.get("agent") or default_agent()
            interrupted_agent = agent["status"] in {"RUNNING", "WAITING_APPROVAL"}
            if interrupted_agent:
                agent["status"] = "ESCALATED"
                agent["pending_approval"] = None
                agent.pop("_approval_deadline", None)
                agent["notifications"].append({"id": uuid.uuid4().hex, "level": "warning", "title": "Agent 运行已中断",
                    "message": "服务已重启；旧审批全部失效，没有自动重新执行动作，请检查记录后重新启动。",
                    "acknowledged": False, "clock": sim.clock})
                if state != "ACTION_EXECUTED":
                    state = "UNRESOLVED"
            if state in {"AWAITING_CONFIRMATION", "VERIFYING"}:
                state = "UNRESOLVED"
            elif state == "DIAGNOSED" and diagnosis_clock != sim.clock:
                stats = detect(sim.window(sim.clock, 30)) if sim.clock else {"alert": False}
                state = "ALERTED" if stats["alert"] else "OBSERVING"
            migrated = state != saved["state"] or interrupted_agent
            self.sessions[rid] = {
                "sim": sim, "state": state,
                "version": saved["version"] + int(migrated), "diagnosis": saved["diagnosis"], "token": None,
                "prediction": saved.get("prediction"), "optimization": saved.get("optimization"),
                "diagnosis_clock": diagnosis_clock, "has_execution": bool(saved.get("has_execution") or executed),
                "created_at": saved.get("created_at", created or 0),
                "agent": agent,
            }
            if migrated:
                self._persist(rid)
        self.db.commit()

    def _persist(self, rid: str):
        s = self.sessions[rid]
        data = {k: v for k, v in s.items() if k not in {"sim", "token"}}
        data["sim"] = s["sim"].export_state()
        self.db.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?)", (rid, json.dumps(data, ensure_ascii=False)))

    def _event(self, rid: str, event: str, detail: dict):
        self.db.execute("INSERT INTO audit VALUES (?, ?, ?, ?)", (rid, event, json.dumps(detail, ensure_ascii=False), time.time()))

    def _get(self, rid):
        if rid not in self.sessions:
            raise SessionNotFound("模拟会话不存在，请创建新会话")
        return self.sessions[rid]

    def _state(self, s):
        # Expiration is derived on reads, without persisting every status poll.
        agent = s["agent"]
        if agent["status"] == "WAITING_APPROVAL":
            return "UNRESOLVED" if approval_expired(agent) else "AWAITING_CONFIRMATION"
        permit = s["token"]
        if s["state"] == "AWAITING_CONFIRMATION" and (not permit or time.monotonic() > permit["expires"]):
            return "UNRESOLVED"
        return s["state"]

    def _ensure_manual(self, s):
        if approval_expired(s["agent"]):
            rid = next(rid for rid, existing in self.sessions.items() if existing is s)
            runner = AgentRunner(self, rid, s["agent"]["run_id"], s["version"])
            runner._escalate_locked(s, "审批已过期且未执行，已结束 Agent 审批，转入手动处理。")
        if active(s["agent"]):
            raise ValueError("Agent 正在运行或等待审批；请通过 Agent 审批，或先停止 Agent 后手动操作")

    @locked
    def create(self, seed: int, scenario: str) -> str:
        sim = Simulator(seed, scenario)
        rid = uuid.uuid4().hex
        self.sessions[rid] = {
            "sim": sim, "state": "OBSERVING", "version": 0, "diagnosis": None, "token": None,
            "prediction": None, "optimization": None, "diagnosis_clock": None,
            "has_execution": False, "created_at": time.time(),
            "agent": default_agent(),
        }
        self._event(rid, "created", {"seed": seed, "scenario": scenario, "simulation": True})
        self._persist(rid)
        return rid

    @locked
    def list_sessions(self):
        recent = sorted(self.sessions.items(), key=lambda pair: pair[1]["created_at"], reverse=True)[:20]
        return [{"run_id": rid, "scenario": s["sim"].scenario, "seed": s["sim"].seed,
                 "clock": s["sim"].clock, "state": self._state(s), "version": s["version"],
                 "created_at": s["created_at"]} for rid, s in recent]

    @locked
    def advance(self, rid: str, n: int):
        s = self._get(rid)
        self._ensure_manual(s)
        if s["state"] in {"ACTION_EXECUTED", "VERIFYING"}:
            raise ValueError("请先完成执行后复核，再推进模拟")
        was_resolved = s["state"] == "RESOLVED"
        s["sim"].advance(n)
        s["version"] += 1
        s["token"] = None
        stats = detect(s["sim"].window(s["sim"].clock, 30))
        s["state"] = "ALERTED" if stats["alert"] else ("RESOLVED" if was_resolved else "OBSERVING")
        self._event(rid, "advance", {"clock": s["sim"].clock, "steps": n})
        self._persist(rid)
        return self.status(rid)

    def _capabilities(self, s, stats):
        ready = s["sim"].clock >= 30
        state = self._state(s)
        awaiting_verification = state in {"ACTION_EXECUTED", "VERIFYING"}
        action_ready = (state == "DIAGNOSED" and s["diagnosis_clock"] == s["sim"].clock
                        and stats.get("quality") == "ok" and stats.get("pressure_mean", 0) > 1.17
                        and stats.get("flow_mean", 1) < .84)
        reasons = {
            "advance": "请先完成执行后复核" if awaiting_verification else "",
            "diagnose": "请先完成执行后复核" if awaiting_verification else ("需要至少 30 分钟观测" if not ready else ""),
            "request_action": "" if action_ready else "需要当前窗口的诊断、合格数据质量以及压力升高且流量降低的模拟异常",
            "verify": "" if state == "ACTION_EXECUTED" else "请先确认并执行模拟处置",
            "optimize": "" if state == "RESOLVED" else "仅复核恢复后允许参数实验",
            "predict": ("训练数据不包含处置后的轨迹，无法评估适用性" if s["has_execution"] else
                        "需要至少 30 分钟观测" if not ready else
                        "预测需要完整的最近 30 分钟观测" if stats.get("quality") != "ok" or stats.get("completeness", 1) < 1 else
                        "请先运行训练命令" if not MODEL_PATH.is_file() else ""),
        }
        if active(s["agent"]):
            reasons = {key: "Agent 正在管理本次流程，请通过 Agent 审批或先停止运行" for key in reasons}
        return {key: not value for key, value in reasons.items()}, reasons

    @locked
    def status(self, rid: str):
        s = self._get(rid)
        sim = s["sim"]
        window = sim.window(sim.clock, 30) if sim.clock else []
        stats = detect(window)
        capabilities, reasons = self._capabilities(s, stats)
        state = self._state(s)
        return {"run_id": rid, "clock": sim.clock, "state": state, "version": s["version"],
                "scenario": sim.scenario, "seed": sim.seed, "created_at": s["created_at"],
                "observations": window, "detection": stats, "simulation": True,
                "diagnosis": s["diagnosis"], "diagnosis_stale": s["diagnosis"] is not None and s["diagnosis_clock"] != sim.clock,
                "confirmation_expired": s["state"] == "AWAITING_CONFIRMATION" and state == "UNRESOLVED",
                "prediction": s["prediction"], "optimization": s["optimization"],
                "agent": public_agent(s["agent"]),
                "capabilities": capabilities, "capability_reasons": reasons}

    def diagnose(self, rid: str, mode: str = "offline"):
        # Slow optional LLM calls must not block status reads or other sessions.
        with self.lock:
            s = self._get(rid)
            self._ensure_manual(s)
            sim = s["sim"]
            if s["state"] in {"ACTION_EXECUTED", "VERIFYING"}:
                raise ValueError("请先完成执行后复核，再进行诊断")
            if sim.clock < 30:
                raise ValueError("需要至少 30 分钟观测")
            version = s["version"]
            window = sim.window(sim.clock, 30)
        result = diagnose(window, mode)
        return self._save_diagnosis(rid, version, result)

    @locked
    def _save_diagnosis(self, rid: str, version: int, result: dict):
        s = self._get(rid)
        sim = s["sim"]
        if s["version"] != version:
            raise ValueError("诊断期间模拟状态已变化，请重新诊断当前窗口")
        s["diagnosis"] = result
        s["diagnosis_clock"] = sim.clock
        s["token"] = None
        if s["state"] != "RESOLVED":
            s["state"] = "DIAGNOSED"
        s["version"] += 1
        self._event(rid, "diagnosis", {"mode": s["diagnosis"]["mode"], "clock": sim.clock})
        self._persist(rid)
        return s["diagnosis"]

    @locked
    def request_action(self, rid: str):
        s = self._get(rid)
        self._ensure_manual(s)
        sim = s["sim"]
        stats = detect(sim.window(sim.clock, 30)) if sim.clock >= 30 else {"quality": "insufficient"}
        capabilities, reasons = self._capabilities(s, stats)
        if not capabilities["request_action"]:
            self._event(rid, "action_denied", {"reason": "preconditions"})
            raise ValueError(reasons["request_action"])
        token = secrets.token_urlsafe(24)
        s["state"] = "AWAITING_CONFIRMATION"
        s["version"] += 1
        s["token"] = {"value": token, "version": s["version"], "expires": time.monotonic() + 120}
        self._event(rid, "confirmation_requested", {"action": "simulate_clean_blockage", "clock": sim.clock, "version": s["version"]})
        self._persist(rid)
        return token

    @locked
    def confirm_action(self, rid: str, token: str, approve: bool, success: bool = True):
        s = self._get(rid)
        self._ensure_manual(s)
        permit = s["token"]
        if (not permit or not secrets.compare_digest(permit["value"], token)
                or permit["version"] != s["version"] or time.monotonic() > permit["expires"]
                or s["state"] != "AWAITING_CONFIRMATION"):
            self._event(rid, "confirmation_denied", {"reason": "invalid, expired, reused or stale"})
            if permit and time.monotonic() > permit["expires"] and s["state"] == "AWAITING_CONFIRMATION":
                s["token"] = None
                s["state"] = "UNRESOLVED"
                s["version"] += 1
                self._persist(rid)
            raise ValueError("确认已失效、已使用或会话发生变化，请重新诊断后申请")
        s["token"] = None
        s["version"] += 1
        if not approve:
            s["state"] = "UNRESOLVED"
            self._event(rid, "confirmation_refused", {})
            self._persist(rid)
            return
        before = s["sim"].snapshot()
        s["sim"].clean(success)
        s["has_execution"] = True
        s["state"] = "ACTION_EXECUTED"
        s["diagnosis_clock"] = None
        self._event(rid, "action_executed", {"action": "simulate_clean_blockage", "before": before, "simulated_success": success})
        self._persist(rid)

    @locked
    def verify(self, rid: str):
        s = self._get(rid)
        self._ensure_manual(s)
        if s["state"] != "ACTION_EXECUTED":
            raise ValueError("当前没有待复核的模拟处置")
        s["state"] = "VERIFYING"
        before = s["sim"].clock
        s["sim"].advance(30)
        s["version"] += 1
        stats = detect(s["sim"].window(s["sim"].clock, 20))
        s["state"] = "RESOLVED" if stats["quality"] == "ok" and not stats["alert"] else "UNRESOLVED"
        result = {"state": s["state"], "before_clock": before, "after_clock": s["sim"].clock, "detection": stats}
        self._event(rid, "verification", result)
        self._persist(rid)
        return result

    @locked
    def report(self, rid: str):
        s = self._get(rid)
        rows = self.db.execute("SELECT event, detail, ts FROM audit WHERE run_id=? ORDER BY rowid", (rid,)).fetchall()
        return {"status": self.status(rid), "diagnosis": s["diagnosis"], "prediction": s.get("prediction"),
                "optimization": s.get("optimization"),
                "agent": public_agent(s["agent"]),
                "audit": [{"event": a, "detail": json.loads(b), "timestamp": c} for a, b, c in rows],
                "limitation": "全部数据和动作均为模拟，不适用于真实设备。"}

    @locked
    def record_result(self, rid: str, kind: str, result: dict):
        if kind not in {"prediction", "optimization"}:
            raise ValueError("invalid result kind")
        s = self._get(rid)
        s[kind] = result
        self._event(rid, kind, {"clock": s["sim"].clock})
        self._persist(rid)

    def agent_start(self, rid: str, mode: str = "rules", objective: str = "检查模拟供浆状态，取得证据并在人工审批后恢复异常工况",
                    max_cycles: int = 4, observation_steps: int = 180):
        if mode not in {"rules", "llm"} or not 1 <= max_cycles <= 6 or not 30 <= observation_steps <= 180:
            raise ValueError("Agent 模式或运行预算不合法")
        if not isinstance(objective, str) or not objective.strip() or len(objective) > 1000:
            raise ValueError("请提供 1–1000 字的任务目标")
        with self.lock:
            s = self._get(rid)
            if active(s["agent"]):
                raise ValueError("当前 Agent 尚未结束，请先完成审批或停止运行")
            if s["state"] in {"ACTION_EXECUTED", "VERIFYING"}:
                raise ValueError("已有模拟动作等待复核，请先完成复核")
            # First takeover can reuse a manually acquired window. A subsequent
            # bounded run must advance time instead of inspecting it forever.
            target = (s["sim"].clock + observation_steps if s["agent"].get("run_id")
                      else max(s["sim"].clock, observation_steps))
            agent = default_agent()
            agent.update({"status": "RUNNING", "mode": mode, "objective": objective.strip(), "run_id": uuid.uuid4().hex,
                          "max_cycles": max_cycles, "planner_source": mode, "observation_steps": observation_steps,
                          "_observation_target": target})
            s["agent"] = agent
            s["token"] = None
            s["state"] = "OBSERVING"
            runner = AgentRunner(self, rid, agent["run_id"], s["version"])
            runner._trace(s, "start", "启动有界 Agent 任务", {"objective": agent["objective"], "mode": mode, "max_cycles": max_cycles}, "user")
            runner._save(s)
        return runner.run("observe")

    def agent_resume(self, rid: str, approval_id: str, approve: bool, success: bool = True):
        with self.lock:
            s = self._get(rid)
            agent = s["agent"]
            pending = agent.get("pending_approval")
            runner = AgentRunner(self, rid, agent["run_id"], s["version"])
            if (agent["status"] != "WAITING_APPROVAL" or not pending
                    or not isinstance(approval_id, str) or not approval_id.isascii()
                    or not secrets.compare_digest(pending["id"], approval_id)
                    or pending["version"] != s["version"] or approval_expired(agent)):
                self._event(rid, "agent_approval_denied", {"reason": "invalid, reused, expired, stopped or stale"})
                if approval_expired(agent):
                    runner._escalate_locked(s, "审批已过期，动作未执行。请重新启动 Agent 获取当前证据。")
                self.db.commit()
                raise ValueError("Agent 审批已失效、已使用或状态已变化，未执行动作")
            action = pending["action"]
            agent["pending_approval"] = None
            agent.pop("_approval_deadline", None)
            s["token"] = None
            if not approve:
                runner._trace(s, "approval", "人员拒绝模拟动作", {"action": action}, "user")
                runner._escalate_locked(s, "人员拒绝本次模拟动作，已生成本地待办；Agent 不再自动申请新动作。")
                return self.status(rid)
            agent["status"] = "RUNNING"
            s["state"] = "DIAGNOSED"
            runner.action, runner.success = action, success
            runner._trace(s, "approval", "人员批准模拟动作", {"action": action}, "user")
            runner._save(s)
        return runner.run("execute")

    @locked
    def agent_stop(self, rid: str):
        s = self._get(rid)
        agent = s["agent"]
        if agent["status"] not in {"RUNNING", "WAITING_APPROVAL"}:
            return self.status(rid)
        runner = AgentRunner(self, rid, agent["run_id"], s["version"])
        agent["status"] = "STOPPED"
        agent["pending_approval"] = None
        agent.pop("_approval_deadline", None)
        s["token"] = None
        if s["state"] != "ACTION_EXECUTED":
            s["state"] = "UNRESOLVED"
        runner._trace(s, "stop", "人员停止 Agent", {"reason": "所有未使用审批已撤销"}, "user")
        runner._notify(s, "info", "Agent 已停止", "当前任务停止，未使用审批已撤销；已执行动作仍保留在审计记录中。")
        runner._save(s)
        return self.status(rid)

    @locked
    def agent_status(self, rid: str):
        return public_agent(self._get(rid)["agent"])
