"""Small, hand-authored smoke evaluation, not an industrial benchmark."""
from coating_agent.knowledge import search
from coating_agent.service import Service

def run_evaluation():
    queries = [("模拟 供浆 阻力 压力 流量", "K02"), ("压力 传感器 偏差", "K03"),
               ("参数优化 泵速 压力 约束", "K09"), ("数据 缺失 故障", "K04"),
               ("电极 涂布 团聚 堵塞 条纹", "K01")]
    retrieval = []
    for q, expected in queries:
        hits = [x["id"] for x in search(q, 3)]
        rank = hits.index(expected)+1 if expected in hits else None
        retrieval.append({"query": q, "expected": expected, "hits": hits, "rank": rank})
    s = Service(":memory:")
    rid = s.create(42, "blockage"); s.advance(rid, 180); d = s.diagnose(rid)
    token = s.request_action(rid); s.confirm_action(rid, token, True, False); failed = s.verify(rid)
    miss = s.create(42, "missing"); s.advance(miss, 90); md = s.diagnose(miss)
    bias = s.create(42, "sensor_bias"); s.advance(bias, 90); bd = s.diagnose(bias)
    unrelated = "电动车胎压怎么处理"
    return {"retrieval": {"cases": len(queries), "recall_at_3": sum(x["rank"] is not None for x in retrieval)/len(queries),
             "mrr": sum(1/x["rank"] if x["rank"] else 0 for x in retrieval)/len(queries), "details": retrieval},
            "unrelated_query": {"query": unrelated, "hits": search(unrelated, 3), "limitation": "单个手写负例"},
            "cases": {"blockage": {"candidates": [x["cause"] for x in d["candidates"]], "alert": s.status(rid)["detection"]["alert"]},
                      "missing": {"uncertain": "不足" in md["uncertainty"], "candidate_count": len(md["candidates"])},
                      "sensor_bias": {"candidates": [x["cause"] for x in bd["candidates"]], "uncertainty": bd["uncertainty"]},
                      "failed_action": failed},
            "limitation": "少量手工构造的模拟案例，与知识库同作者；不代表真实工业诊断准确率。"}
