"""Train and evaluate on isolated synthetic assets; no real-data claims."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, brier_score_loss
from coating_agent.predict import features, label, split_assets
from coating_agent.simulator import Simulator

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--assets", type=int, default=120)
    parser.add_argument("--output", default="artifacts"); args = parser.parse_args()
    if args.assets < 40: raise ValueError("at least 40 assets")
    from xgboost import XGBClassifier
    splits = split_assets(list(range(args.assets)), args.seed)
    groups = [[], [], []]
    for g, asset_ids in enumerate(splits):
        for asset in asset_ids:
            scenario = "blockage" if asset % 2 == 0 else "normal"
            sim = Simulator(args.seed * 1000 + asset, scenario); sim.advance(360)
            for t in range(30, 331, 5):
                y = label(t, sim.event_time, sim.clock)
                if y is not None: groups[g].append((features(sim.window(t, 30)), y, asset, t, sim.event_time))
    def xy(group): return np.array([r[0] for r in group]), np.array([r[1] for r in group])
    xtrain, ytrain = xy(groups[0]); xval, yval = xy(groups[1]); xtest, ytest = xy(groups[2])
    models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=args.seed, max_iter=1000)),
              "xgboost": XGBClassifier(n_estimators=80, max_depth=3, learning_rate=.08, random_state=args.seed, n_jobs=1)}
    report = {"domain": "仅模拟阻力故障，未来30分钟事件，非RUL", "seed": args.seed,
              "assets": args.assets, "split": {"train": splits[0], "validation": splits[1], "test": splits[2]},
              "test_windows": len(ytest), "test_positive_rate": float(ytest.mean()), "python": platform.python_version(), "models": {}}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    split_json = json.dumps(report["split"], sort_keys=True).encode(); report["split_sha256"] = hashlib.sha256(split_json).hexdigest()
    # A threshold rule is calibrated on validation only and evaluated on the untouched test split.
    for name, model in [("rule", None)] + list(models.items()):
        if model is None:
            val_prob = np.clip((xval[:,0] - 1.0) / .5, 0, 1); prob = np.clip((xtest[:,0] - 1.0) / .5, 0, 1)
        else:
            model.fit(xtrain, ytrain); val_prob = model.predict_proba(xval)[:,1]; prob = model.predict_proba(xtest)[:,1]
            joblib.dump(model, out / f"{name}.joblib")
        thresholds = np.linspace(.05, .95, 19)
        from sklearn.metrics import f1_score
        threshold = float(max(thresholds, key=lambda z: f1_score(yval, val_prob >= z, zero_division=0)))
        report["models"][name] = {"threshold_from_validation": threshold, "pr_auc": float(average_precision_score(ytest, prob)),
          "precision": float(precision_score(ytest, prob >= threshold, zero_division=0)),
          "recall": float(recall_score(ytest, prob >= threshold, zero_division=0)), "brier": float(brier_score_loss(ytest, prob))}
        events = {}; normal_runs = set(); false_alarm_runs = set()
        for row, score in zip(groups[2], prob):
            _, _, asset, t, event = row
            if event is None:
                normal_runs.add(asset)
                if score >= threshold: false_alarm_runs.add(asset)
            elif score >= threshold and 0 < event-t <= 30:
                events[asset] = max(events.get(asset, 0), event-t)
        event_assets = {r[2] for r in groups[2] if r[4] is not None}
        report["models"][name].update({"event_recall": len(events)/len(event_assets) if event_assets else None,
             "mean_first_warning_lead_minutes": float(np.mean(list(events.values()))) if events else None,
             "normal_trajectory_false_alarm_rate": len(false_alarm_runs)/len(normal_runs) if normal_runs else None})
    # Explicit out-of-domain checks: reporting trigger rate, not classification accuracy.
    model = models["xgboost"]; threshold = report["models"]["xgboost"]["threshold_from_validation"]
    report["out_of_domain"] = {}
    for scenario in ("sensor_bias", "pump_loss", "mixed", "shifted_blockage"):
        scores = []
        for asset in splits[2]:
            sim = Simulator(args.seed * 2000 + asset, "blockage" if scenario == "shifted_blockage" else scenario)
            if scenario == "shifted_blockage": sim.a *= 1.5; sim.b *= .75
            sim.advance(180)
            scores.append(float(model.predict_proba([features(sim.window(180, 30))])[0][1]))
        report["out_of_domain"][scenario] = {"runs": len(scores), "score_at_180_over_threshold": sum(x >= threshold for x in scores)/len(scores),
                                               "interpretation": "只报告触发比例，不将此视为诊断准确率"}
    (out / "model-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test_windows": report["test_windows"], "test_positive_rate": report["test_positive_rate"], "models": report["models"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
