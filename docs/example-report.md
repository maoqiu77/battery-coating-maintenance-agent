# 示例报告：模拟供浆阻力异常

本报告由固定种子 42 的模拟器生成；数据和动作均为模拟，不代表真实设备。

## 模拟诊断与证据

```json
{
  "mode": "规则模板",
  "evidence": [
    {
      "id": "O1",
      "kind": "模拟观测",
      "data": {
        "alert": true,
        "quality": "ok",
        "pressure_mean": 1.4504,
        "flow_mean": 0.5388,
        "reason": "模拟阈值异常"
      }
    },
    {
      "id": "K03",
      "kind": "项目假设",
      "source": "docs/superpowers/specs/2026-10-27-coating-supply-agent-design.md",
      "text": "模拟压力传感器偏差可能使压力读数上升，即使流量稳定；需检查传感器。"
    },
    {
      "id": "K10",
      "kind": "模拟规则",
      "source": "docs/superpowers/specs/2026-10-27-coating-supply-agent-design.md",
      "text": "诊断必须标注观测和知识来源，压力流量不足以唯一确定设备故障。"
    },
    {
      "id": "K02",
      "kind": "项目假设",
      "source": "docs/superpowers/specs/2026-10-27-coating-supply-agent-design.md",
      "text": "本模拟器中阻力增长可导致压力上升且流量下降；不代表实际设备因果关系。"
    },
    {
      "id": "K07",
      "kind": "项目假设",
      "source": "docs/superpowers/specs/2026-10-27-coating-supply-agent-design.md",
      "text": "泵速、流量、压力关系是无量纲简化模型，不应用于生产配方。"
    },
    {
      "id": "K08",
      "kind": "项目假设",
      "source": "docs/superpowers/specs/2026-10-27-coating-supply-agent-design.md",
      "text": "模拟泵效率下降可使流量降低，需要检查泵和压力读数。"
    }
  ],
  "candidates": [
    {
      "cause": "模拟供浆阻力上升",
      "evidence_ids": [
        "O1",
        "K02"
      ],
      "missing": "缺少独立流路检查及传感器校验"
    },
    {
      "cause": "压力传感器偏差或其他原因",
      "evidence_ids": [
        "O1",
        "K03"
      ],
      "missing": "缺少独立压力测量"
    }
  ],
  "uncertainty": "模拟证据不足以唯一确定原因；需人工检查。",
  "next_step": "检查观测质量并由人员审核模拟处置",
  "model_calls": 0
}
```

## 复核

```json
{
  "state": "RESOLVED",
  "before_clock": 180,
  "after_clock": 210,
  "detection": {
    "alert": false,
    "quality": "ok",
    "pressure_mean": 1.058,
    "flow_mean": 0.9172,
    "reason": "模拟状态正常"
  }
}
```

## 审计事件

- created
- advance
- diagnosis
- confirmation_requested
- action_executed
- verification

结果只说明此模拟模型在上述输入与设定下的响应。
