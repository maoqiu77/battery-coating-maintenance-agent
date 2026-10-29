import json
from pathlib import Path
from coating_agent.evaluation import run_evaluation

if __name__ == "__main__":
    report = run_evaluation(); Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
