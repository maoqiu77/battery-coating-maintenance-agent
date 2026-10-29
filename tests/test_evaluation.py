from coating_agent.evaluation import run_evaluation

def test_eval_reports_failures_and_provenance():
    report = run_evaluation()
    assert report["retrieval"]["cases"] >= 3
    assert 0 <= report["retrieval"]["recall_at_3"] <= 1
    assert report["cases"]["missing"]["uncertain"]
    assert report["cases"]["failed_action"]["state"] == "UNRESOLVED"
    assert "模拟" in report["limitation"]
