"""Loopback-only demo API. No industrial integrations."""
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from coating_agent.service import MODEL_PATH, Service, SessionNotFound

STATIC_DIR = Path(__file__).with_name("static")
app = FastAPI(title="模拟涂布供浆运维 Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
service = Service()


class Create(BaseModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    scenario: Literal["blockage", "sensor_bias", "missing", "normal", "pump_loss", "mixed"] = "blockage"


class Advance(BaseModel):
    steps: int = Field(default=1, ge=1, le=3600)


class Diagnose(BaseModel):
    mode: Literal["offline", "llm"] = "offline"


class Confirm(BaseModel):
    token: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_-]+$")
    approve: bool
    success: bool = True


class Optimize(BaseModel):
    target: float = Field(default=1.0, ge=.5, le=1.5, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class AgentStart(BaseModel):
    mode: Literal["rules", "llm"] = "rules"
    objective: str = Field(default="检查模拟供浆状态，取得证据并在人工审批后恢复异常工况", min_length=1, max_length=1000)
    max_cycles: int = Field(default=4, ge=1, le=6)
    observation_steps: int = Field(default=180, ge=30, le=180)


class AgentApproval(BaseModel):
    approval_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_-]+$")
    approve: bool
    success: bool = True


def call(fn, *args):
    try:
        return fn(*args)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index():
    return STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config():
    model = os.environ.get("COATING_LLM_MODEL", "").strip()
    return {"simulation": True, "llm_available": bool(model and os.environ.get("OPENAI_API_KEY")),
            "llm_model": model or None, "llm_api": os.environ.get("COATING_LLM_API", "responses"),
            "prediction_available": MODEL_PATH.is_file()}


@app.get("/api/health")
def health():
    return {"status": "ok", **config()}


@app.get("/api/knowledge")
def knowledge(q: str = Query(default="", max_length=500)):
    from coating_agent.knowledge import CARDS, search
    return {"cards": search(q.strip(), 10) if q.strip() else CARDS}


@app.get("/api/sessions")
def sessions():
    return {"sessions": call(service.list_sessions)}


@app.post("/api/sessions")
def create(body: Create):
    return {"run_id": call(service.create, body.seed, body.scenario)}


@app.get("/api/sessions/{rid}")
def status(rid: str):
    return call(service.status, rid)


@app.get("/api/sessions/{rid}/window")
def window(rid: str, end: int = Query(ge=1), size: int = Query(default=30, ge=1, le=3600)):
    with service.lock:
        s = call(service._get, rid)
        return call(s["sim"].window, end, size)


@app.post("/api/sessions/{rid}/advance")
def advance(rid: str, body: Advance):
    return call(service.advance, rid, body.steps)


@app.post("/api/sessions/{rid}/diagnose")
def diagnose(rid: str, body: Diagnose):
    return call(service.diagnose, rid, body.mode)


@app.post("/api/sessions/{rid}/request-action")
def request_action(rid: str):
    return {"token": call(service.request_action, rid), "expires_seconds": 120, "action": "simulate_clean_blockage"}


@app.post("/api/sessions/{rid}/confirm")
def confirm(rid: str, body: Confirm):
    call(service.confirm_action, rid, body.token, body.approve, body.success)
    return call(service.status, rid)


@app.post("/api/sessions/{rid}/verify")
def verify(rid: str):
    return call(service.verify, rid)


@app.post("/api/sessions/{rid}/agent/start")
def agent_start(rid: str, body: AgentStart):
    return call(service.agent_start, rid, body.mode, body.objective, body.max_cycles, body.observation_steps)


@app.post("/api/sessions/{rid}/agent/resume")
def agent_resume(rid: str, body: AgentApproval):
    return call(service.agent_resume, rid, body.approval_id, body.approve, body.success)


@app.post("/api/sessions/{rid}/agent/stop")
def agent_stop(rid: str):
    return call(service.agent_stop, rid)


@app.get("/api/sessions/{rid}/agent")
def agent_status(rid: str):
    return call(service.agent_status, rid)


@lru_cache(maxsize=2)
def prediction_model(path: str, modified_ns: int):
    import joblib
    return joblib.load(path)


@app.get("/api/sessions/{rid}/prediction")
def prediction(rid: str):
    from coating_agent.predict import features
    with service.lock:
        s = call(service._get, rid)
        sim = s["sim"]
        state = call(service.status, rid)
        if not state["capabilities"]["predict"]:
            return {"available": False, "reason": state["capability_reasons"]["predict"]}
        rows = sim.window(sim.clock, 30)
        if any(row["pressure"] is None or row["flow"] is None for row in rows):
            return {"available": False, "reason": "预测需要完整的最近 30 分钟观测"}
        x = call(features, rows)
        try:
            model = prediction_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
            score = float(model.predict_proba([x])[0][1])
        except (OSError, ImportError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="预测模型暂不可用，请重新训练并检查本地依赖") from exc
        result = {"available": True, "score": score, "model": "xgboost", "clock": sim.clock,
                  "target": "未来30分钟内模拟阻力故障事件",
                  "limitation": "仅模拟训练域，非真实风险或剩余寿命；训练仅含故障事件前窗口，当前观测无法确认是否越过事件，可能不适用。"}
        call(service.record_result, rid, "prediction", result)
        return result


@app.post("/api/sessions/{rid}/optimize")
def optimize(rid: str, body: Optimize):
    from coating_agent.optimize import optimize as run
    with service.lock:
        s = call(service._get, rid)
        call(service._ensure_manual, s)
        if s["state"] != "RESOLVED":
            raise HTTPException(status_code=400, detail="仅恢复后允许离线模拟优化")
        result = call(run, s["sim"], body.target, body.seed)
        result["clock"] = s["sim"].clock
        call(service.record_result, rid, "optimization", result)
        return result


@app.get("/api/sessions/{rid}/report")
def report(rid: str):
    return call(service.report, rid)


@app.get("/api/sessions/{rid}/report.md", response_class=PlainTextResponse)
def report_markdown(rid: str):
    import json
    r = call(service.report, rid)
    return ("# 模拟运维报告\n\n" + r["limitation"] + "\n\n## 当前状态\n\n```json\n" +
            json.dumps(r["status"], ensure_ascii=False, indent=2) + "\n```\n\n## 诊断与证据\n\n```json\n" +
            json.dumps(r["diagnosis"], ensure_ascii=False, indent=2) + "\n```\n\n## 模拟预测和参数实验\n\n```json\n" +
            json.dumps({"prediction": r["prediction"], "optimization": r["optimization"]}, ensure_ascii=False, indent=2) +
            "\n```\n\n## Agent 决策、工具与复核轨迹\n\n```json\n" +
            json.dumps(r["agent"], ensure_ascii=False, indent=2) +
            "\n```\n\n## 审计记录\n\n```json\n" +
            json.dumps(r["audit"], ensure_ascii=False, indent=2) + "\n```\n")
