"""Evidence-constrained LangGraph; the optional LLM has read-only tools only."""
from typing import TypedDict
import json
import os
from langgraph.graph import StateGraph, START, END
from coating_agent.detection import detect
from coating_agent.knowledge import search

class State(TypedDict, total=False):
    window: list[dict]
    mode: str
    evidence: list[dict]
    result: dict

def gather(state: State) -> dict:
    stats = detect(state["window"])
    queries = {
        "pressure_high_flow_low": ("模拟 供浆 阻力 压力 流量", "压力 传感器 偏差"),
        "pressure_high": ("压力 传感器 偏差", "模拟 供浆 阻力 压力 流量"),
        "flow_low": ("泵 效率 下降 流量", "模拟 供浆 阻力 压力 流量"),
        "normal": ("供浆 模型 边界", "涂布 证据 不确定性"),
        "insufficient_data": ("数据 缺失 故障",),
    }
    cards = {}
    for query in queries[stats["signal_pattern"]]:
        for card in search(query, 3):
            cards.setdefault(card["id"], card)
    evidence = [{"id": "O1", "kind": "模拟观测", "data": stats}]
    evidence += [dict(c) for c in cards.values()]
    return {"evidence": evidence}

def validate_citations(result: dict) -> bool:
    if not isinstance(result, dict) or not isinstance(result.get("evidence"), list) or not isinstance(result.get("candidates"), list):
        return False
    evidence = result["evidence"]
    if not all(isinstance(e, dict) and isinstance(e.get("id"), str) for e in evidence):
        return False
    ids = {e["id"] for e in evidence}
    return all(isinstance(c, dict) and isinstance(c.get("evidence_ids"), list) and bool(c["evidence_ids"])
               and all(isinstance(i, str) and i in ids for i in c["evidence_ids"])
               for c in result["candidates"])

def validate_llm_result(result: dict, accessed: set[str], used_statistics: bool) -> bool:
    if not isinstance(result, dict):
        return False
    candidates = result.get("candidates")
    if not used_statistics or not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
        return False
    if not all(isinstance(result.get(k), str) and result[k].strip() for k in ("uncertainty", "next_step")):
        return False
    return all(isinstance(c, dict) and all(isinstance(c.get(k), str) and c[k].strip() for k in ("cause", "missing"))
               and isinstance(c.get("evidence_ids"), list) and bool(c["evidence_ids"])
               and "O1" in c["evidence_ids"] and any(i.startswith("K") for i in c["evidence_ids"] if isinstance(i, str))
               and all(isinstance(i, str) and i in accessed for i in c["evidence_ids"]) for c in candidates)

def template(state: State) -> dict:
    stats = state["evidence"][0]["data"]
    ids = {x["id"] for x in state["evidence"]}
    pattern = stats["signal_pattern"]
    alternatives = {
        "pressure_high_flow_low": [
            ("模拟供浆阻力上升", "K02", "缺少独立流路检查及传感器校验"),
            ("压力传感器偏差与流量异常并存", "K03", "缺少独立压力测量，传感器偏差本身不能解释流量偏低"),
        ],
        "pressure_high": [
            ("压力传感器偏差", "K03", "缺少独立压力测量，尚不能确认偏差"),
            ("供浆阻力变化或其他工况扰动", "K02", "当前未见同步低流量信号，阻力解释证据较弱"),
        ],
        "flow_low": [
            ("模拟泵效率下降", "K08", "缺少独立泵性能检查及流量测量"),
            ("供浆流路异常或其他工况扰动", "K02", "当前未见同步高压力信号，需要更多流路观测"),
        ],
    }
    candidates = [{"cause": cause, "evidence_ids": ["O1", card], "missing": missing}
                  for cause, card, missing in alternatives.get(pattern, []) if card in ids]
    summaries = {
        "pressure_high_flow_low": "压力偏高、流量偏低，优先核查模拟供浆阻力变化。",
        "pressure_high": "压力单独偏高，优先核查模拟压力传感器偏差。",
        "flow_low": "流量单独偏低，优先核查模拟泵效率及供浆状态。",
        "normal": "当前观测未触发模拟异常阈值。",
        "insufficient_data": "有效观测不足，暂不判断故障原因。",
    }
    steps = {
        "pressure_high_flow_low": "核对观测与知识证据；如申请模拟清理，需人工确认并执行后复核。",
        "pressure_high": "在模拟分析中对照独立压力读数与流量变化，暂不建议清理。",
        "flow_low": "在模拟分析中核查泵效率与流量观测，暂不建议清理。",
        "normal": "继续采集模拟观测，关注压力与流量是否发生同步变化。",
        "insufficient_data": "优先补齐有效观测并检查采集链路，数据质量达标后重新诊断。",
    }
    uncertainty = ("数据不足，无法判断原因" if stats["quality"] != "ok" else
                   "当前窗口未见阈值异常，不代表后续不会出现故障。" if pattern == "normal" else
                   "模拟证据不足以唯一确定原因；需人工检查。")
    return {"mode": "规则模板", "evidence": state["evidence"], "candidates": candidates,
            "summary": summaries[pattern], "severity": stats["severity"], "signal_pattern": pattern,
            "uncertainty": uncertainty, "next_step": steps[pattern], "model_calls": 0}

def llm_diagnosis(state: State) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    accessed = {}
    @tool
    def get_statistics() -> str:
        """Get the current simulation window statistics and evidence ID."""
        accessed["O1"] = state["evidence"][0]
        return json.dumps(state["evidence"][0], ensure_ascii=False)

    @tool
    def search_knowledge(query: str) -> str:
        """Search evidence cards about simulated coating maintenance."""
        cards = search(query, 5)
        for card in cards:
            accessed[card["id"]] = dict(card)
        return json.dumps(cards, ensure_ascii=False)

    model_name = os.environ.get("COATING_LLM_MODEL", "")
    if not model_name or not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("missing model or API key")
    llm = ChatOpenAI(model=model_name, base_url=os.environ.get("OPENAI_BASE_URL") or None, timeout=20, max_retries=0).bind_tools([get_statistics, search_knowledge])
    messages = [SystemMessage(content="你是模拟设备诊断助手。只使用工具取证，最多调用工具两轮。最终只输出 JSON：candidates 为包含 cause、evidence_ids、missing 的数组，uncertainty、next_step 为字符串。只引用取证返回的 ID，不要给出现实设备操作建议。"), HumanMessage(content="诊断当前模拟供浆工况。")]
    for turn in range(3):
        response = llm.invoke(messages); messages.append(response)
        if not response.tool_calls:
            data = json.loads(response.content)
            if not validate_llm_result(data, set(accessed), "O1" in accessed):
                raise ValueError("model must cite actually retrieved evidence")
            baseline = template(state)
            result = {"mode": "LLM", "evidence": list(accessed.values()), "candidates": data["candidates"],
                      "summary": baseline["summary"], "severity": baseline["severity"],
                      "signal_pattern": baseline["signal_pattern"],
                      "uncertainty": data["uncertainty"], "next_step": data["next_step"], "model_calls": turn + 1}
            if not isinstance(result["candidates"], list) or not validate_citations(result) or not result["uncertainty"]:
                raise ValueError("invalid evidence references")
            return result
        if turn == 2 or len(response.tool_calls) > 4:
            raise ValueError("tool budget exceeded")
        for call in response.tool_calls:
            if call["name"] == "get_statistics": output = get_statistics.invoke(call["args"])
            elif call["name"] == "search_knowledge": output = search_knowledge.invoke(call["args"])
            else: raise ValueError("unknown tool")
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))
    raise ValueError("no result")

def respond(state: State) -> dict:
    stats = state["evidence"][0]["data"]
    if state["mode"] == "llm" and stats["quality"] == "ok" and stats["alert"]:
        try: return {"result": llm_diagnosis(state)}
        except Exception as exc:
            result = template(state); result["warning"] = f"LLM 不可用或输出未通过校验（{type(exc).__name__}），已回退规则模板"
            return {"result": result}
    result = template(state)
    if state["mode"] == "llm":
        result["warning"] = ("有效观测不足，未调用 LLM，优先处理数据质量。" if stats["quality"] != "ok" else
                             "当前窗口未触发模拟异常，已采用规则检查结果，未调用 LLM。")
    return {"result": result}

graph = StateGraph(State)
graph.add_node("gather", gather); graph.add_node("respond", respond)
graph.add_edge(START, "gather"); graph.add_edge("gather", "respond"); graph.add_edge("respond", END)
workflow = graph.compile()

def diagnose(window: list[dict], mode: str = "offline") -> dict:
    if mode not in {"offline", "llm"}: raise ValueError("invalid mode")
    return workflow.invoke({"window": window, "mode": mode})["result"]
