# FailureLens

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

**AI 辅助故障定位与开发者工具集。** FailureLens 根据故障日志（堆栈跟踪）和源代码仓库，
对可能导致故障的代码位置进行排序，并展示每个候选位置的**证据**，
帮助开发者、CI 系统和 AI 编程代理调查根本原因。

[![CI](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml/badge.svg)](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](docs/installation.md)

本 README 是[英文版](README.md)的简体中文译本。详细文档目前以英文提供。

> **当前状态。** FailureLens 提供**可运行、依赖精简的基线方案**
> （堆栈跟踪与词项匹配，仅需 CPU，无需网络）。
> **实验性的已训练编码器**已发布至
> [LevelCapTech/failurelens](https://huggingface.co/LevelCapTech/failurelens)。
> 独立留出集评估尚未进行，目前没有证据表明其优于基线方案或基础编码器。
> 请使用模型卡中直接调用 SentenceTransformers 的独立示例：
> 库中通用的 `from_pretrained` 加载方式尚不支持该已训练模型仓库的自定义代码和查询转换。

## 为什么使用 FailureLens？

堆栈跟踪告诉你的是*错误在哪里显现*，不一定是*错误由什么引起*。
FailureLens 将两者区分开来，对候选故障位置进行排序，并且仅附上**实际观察到的证据**：
对应哪个堆栈帧、是否为异常发生位置、哪些错误词项匹配等，方便你判断每条建议。
它不是 LLM API 的封装、通用代码搜索工具或错误信息摘要工具。

## 功能

- 🔦 **故障定位基线方案** — 使用透明、确定性的启发式方法，对文件、函数和行范围进行排序。
  无需 GPU、网络连接或模型下载。
- 🧩 **统一的结果格式** — Python API 与 CLI 的 JSON 输出共享同一套带版本号的数据结构。
- 🧪 **如实呈现结果** — `score_type` 明确标示启发式分数，而非概率；
  未知位置以 `null` 表示，绝不编造；状态字段区分“无法得出结论”和“处理失败”。
- 🪶 **轻量核心** — 必需依赖只有 `pydantic` 和 `typer`。机器学习功能通过可选依赖提供。
- 🤖 **便于代理集成** — 稳定的退出码和无额外输出的 JSON，方便自动化处理。
- 📏 **可衡量** — 提供可复用的排序评估指标和可运行的评估工具。

## 安装

需要 Python **3.11 或更高版本**。目前尚未发布到 PyPI，请从源码安装：

```bash
git clone https://github.com/LevelCapTech/FailureLens.git
cd FailureLens
pip install -e "packages/python"          # 仅安装轻量基线方案
# pip install -e "packages/python[ml]"    # 添加可选的嵌入后端
```

可选依赖及故障排查说明见[安装指南](docs/installation.md)。

## 快速开始

```bash
failurelens diagnose \
  --repo examples/fixtures/sample_project \
  --log  examples/fixtures/sample_project/failure.log \
  --top 5
```

```python
from failurelens import FailureLens

lens = FailureLens.from_baseline()
result = lens.diagnose(
    repository="examples/fixtures/sample_project",
    error_log="examples/fixtures/sample_project/failure.log",
    top_k=5,
)
for c in result.candidates:
    print(
        f"#{c.rank} {c.file}:{c.start_line}-{c.end_line} {c.function} "
        f"score={c.score:.3f} ({c.score_type})"
    )
    for ev in c.evidence:
        print("   -", ev.description)
```

随附示例展示了一个关键区别：`ZeroDivisionError` 在 `divide()` 中**显现**，
但**根本原因**在于 `average()` 没有检查空输入。输出的证据说明了每个候选位置被选中的原因。

## 仓库结构

```text
packages/python/   可安装的库与 failurelens CLI（产品主体）
examples/          可运行的演示与示例数据
evaluation/        基于实测的基准评估工具、指标文档和报告
datasets/          数据记录格式与流水线设计（不包含大型或虚构数据）
models/            模型设计（权重单独托管，不存入 Git）
training/          实验性 GPU 微调与 CPU 小型演示流水线（按需运行）
docs/              完整规格文档（从 docs/architecture.md 开始）
AGENTS.md          贡献者与 AI 代理的统一工作规范
```

## 文档

以下详细文档以英文提供。

| 主题 | 文档 |
|------|------|
| 工作原理 | [docs/architecture.md](docs/architecture.md) |
| 安装 | [docs/installation.md](docs/installation.md) |
| 首次运行 | [docs/getting-started.md](docs/getting-started.md) |
| Python API 与数据结构 | [docs/api.md](docs/api.md) |
| CLI 与退出码 | [docs/cli.md](docs/cli.md) |
| 数据集 | [docs/datasets.md](docs/datasets.md) |
| 训练 | [docs/training.md](docs/training.md) |
| 评估 | [docs/evaluation.md](docs/evaluation.md) |
| 安全 | [docs/security.md](docs/security.md) · [SECURITY.md](SECURITY.md) |
| 模型卡 | [docs/model-card.md](docs/model-card.md) |
| 路线图 | [docs/roadmap.md](docs/roadmap.md) |
| 需求 | [docs/requirements.md](docs/requirements.md) |
| AI 模型设计决策关卡 | [docs/decision-gate.md](docs/decision-gate.md) · [数据契约](docs/data-contract.md) · [评估协议](docs/evaluation-protocol.md) |
| 贡献指南 | [CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) |

## 开发

```bash
make install     # 以可编辑模式安装，并包含开发工具
make check       # ruff 检查与格式检查、mypy 严格类型检查、pytest（与 CI 一致）
make demo        # 在随附示例上运行基线方案
make build       # 构建 sdist 和 wheel
python evaluation/run_baseline_eval.py   # 实测基线评估指标
```

## 参与贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和
[AGENTS.md](AGENTS.md)（工作规范，尤其是如实呈现结果的规则），
并遵守[行为准则](CODE_OF_CONDUCT.md)。

## 许可证

本仓库的源代码采用 Apache-2.0 许可证，详见 [LICENSE](LICENSE)。
已发布的实验模型权重采用 MIT 许可证，并保留上游代码的权利声明；
详见[模型卡](docs/model-card.md)。
