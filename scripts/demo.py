"""Generate a reproducible, clearly simulated Markdown report."""
import json
from pathlib import Path
from coating_agent.service import Service

s = Service(":memory:")
rid = s.create(42, "blockage"); s.advance(rid, 180); diagnosis = s.diagnose(rid)
token = s.request_action(rid); s.confirm_action(rid, token, True); verification = s.verify(rid)
report = s.report(rid)
text = ("# 示例报告：模拟供浆阻力异常\n\n"
        "本报告由固定种子 42 的模拟器生成；数据和动作均为模拟，不代表真实设备。\n\n"
        "## 模拟诊断与证据\n\n```json\n" + json.dumps(diagnosis, ensure_ascii=False, indent=2) +
        "\n```\n\n## 复核\n\n```json\n" + json.dumps(verification, ensure_ascii=False, indent=2) +
        "\n```\n\n## 审计事件\n\n" + "\n".join(f"- {item['event']}" for item in report["audit"]) +
        "\n\n结果只说明此模拟模型在上述输入与设定下的响应。\n")
Path("docs/example-report.md").write_text(text, encoding="utf-8")
print("Wrote docs/example-report.md (synthetic data)")
