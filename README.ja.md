# FailureLens

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

**AIを活用した不具合の原因箇所の特定と、開発者向けツールキット。**
FailureLensは、障害ログ（スタックトレース）とソースリポジトリをもとに、
原因と考えられるコードの候補箇所を順位付けし、それぞれの**根拠**を示します。
開発者、CIシステム、AIコーディングエージェントが根本原因の調査に取り組むためのツールです。

[![CI](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml/badge.svg)](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](docs/installation.md)

このREADMEは[英語版](README.md)の日本語訳です。詳細ドキュメントは英語で提供されています。

> **現在の状況。** FailureLensは、**動作する軽量なベースライン**を提供しています
> （スタックトレースと字句一致を利用、CPUのみ、ネットワーク不要）。
> **実験段階の学習済みエンコーダー**を
> [LevelCapTech/failurelens](https://huggingface.co/LevelCapTech/failurelens)で公開しています。
> 独立したホールドアウト評価は未実施で、ベースラインやベースエンコーダーに対する優位性は確認されていません。
> モデルカードにある、SentenceTransformersを直接使う実行例をご利用ください。
> ライブラリの汎用的な`from_pretrained`経由では、この学習済みモデルリポジトリの
> カスタムコードとクエリ変換にまだ対応していません。

## FailureLensを使う理由

スタックトレースが示すのは、必ずしも原因そのものではなく、*エラーが表面化した場所*です。
FailureLensは両者を区別し、原因箇所の候補を順位付けします。
どのフレームに現れたか、例外の発生箇所か、どのエラートークンが一致したかといった、
**実際に観測した根拠**だけを添えるため、各候補の妥当性を判断できます。
LLM APIのラッパー、汎用コード検索、エラーメッセージの要約ツールではありません。

## 主な機能

- 🔦 **原因箇所を特定するベースライン** — 仕組みが明確で決定的なヒューリスティックにより、
  ファイル・関数・行範囲を順位付けします。GPU、ネットワーク、モデルのダウンロードは不要です。
- 🧩 **共通の出力形式** — Python APIとCLIのJSON出力で、同じバージョン管理されたスキーマを使います。
- 🧪 **誤解を招かない設計** — `score_type`はスコアが確率ではなくヒューリスティックであることを明示します。
  不明な位置は推測せず`null`で返し、判定不能と処理失敗をステータスで区別します。
- 🪶 **軽量なコア** — 必須の依存関係は`pydantic`と`typer`のみ。機械学習機能は任意で追加できます。
- 🤖 **エージェントとの連携** — 安定した終了コードと、余分な出力のないJSONで自動化に対応します。
- 📏 **測定可能** — 再利用できるランキング評価指標と、実行可能な評価ハーネスを備えています。

## インストール

Python **3.11以上**が必要です。PyPIには未公開のため、ソースからインストールしてください。

```bash
git clone https://github.com/LevelCapTech/FailureLens.git
cd FailureLens
pip install -e "packages/python"          # ベースラインのみ（軽量）
# pip install -e "packages/python[ml]"    # 任意の埋め込みバックエンドを追加
```

追加機能のインストールやトラブルシューティングは、[インストールガイド](docs/installation.md)をご覧ください。

## クイックスタート

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

同梱のサンプルでは、`ZeroDivisionError`が**表面化する**のは`divide()`ですが、
**根本原因**は空の入力をチェックしていない`average()`です。
各候補が選ばれた理由は、出力の根拠から確認できます。

## リポジトリ構成

```text
packages/python/   インストール可能なライブラリとfailurelens CLI（製品本体）
examples/          実行可能なデモとサンプルデータ
evaluation/        実測に基づくベンチマークハーネス、評価指標の説明、レポート
datasets/          データレコードのスキーマとパイプライン設計（巨大・架空のデータは含まない）
models/            モデル設計（重みは別途配布し、Gitには格納しない）
training/          実験用GPUファインチューニングとCPU用の小規模パイプライン（任意実行）
docs/              詳細仕様（docs/architecture.mdから参照）
AGENTS.md          コントリビューターとAIエージェント向けの作業規約
```

## ドキュメント

以下の詳細ドキュメントは英語です。

| トピック | ドキュメント |
|----------|--------------|
| 仕組み | [docs/architecture.md](docs/architecture.md) |
| インストール | [docs/installation.md](docs/installation.md) |
| 初回実行 | [docs/getting-started.md](docs/getting-started.md) |
| Python APIとスキーマ | [docs/api.md](docs/api.md) |
| CLIと終了コード | [docs/cli.md](docs/cli.md) |
| データセット | [docs/datasets.md](docs/datasets.md) |
| 学習 | [docs/training.md](docs/training.md) |
| 評価 | [docs/evaluation.md](docs/evaluation.md) |
| セキュリティ | [docs/security.md](docs/security.md) · [SECURITY.md](SECURITY.md) |
| モデルカード | [docs/model-card.md](docs/model-card.md) |
| ロードマップ | [docs/roadmap.md](docs/roadmap.md) |
| 要件 | [docs/requirements.md](docs/requirements.md) |
| AIモデルの設計判断ゲート | [docs/decision-gate.md](docs/decision-gate.md) · [データ契約](docs/data-contract.md) · [評価プロトコル](docs/evaluation-protocol.md) |
| コントリビューション | [CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) |

## 開発

```bash
make install     # 開発ツールを含めて編集可能な状態でインストール
make check       # ruff lint・整形チェック、mypy（strict）、pytest（CIと同じ）
make demo        # 同梱のサンプルでベースラインを実行
make build       # sdistとwheelをビルド
python evaluation/run_baseline_eval.py   # ベースラインの評価指標を実測
```

## コントリビューション

コントリビューションを歓迎します。[CONTRIBUTING.md](CONTRIBUTING.md)と
[AGENTS.md](AGENTS.md)（作業規約、特に事実に忠実であるためのルール）を読み、
[行動規範](CODE_OF_CONDUCT.md)に従ってください。

## ライセンス

このリポジトリのソースコードにはApache-2.0が適用されます。[LICENSE](LICENSE)をご覧ください。
公開済みの実験モデルの重みはMITライセンスで、上流コードの権利表示を保持しています。
詳しくは[モデルカード](docs/model-card.md)をご覧ください。
