# Gate A — Go/No-Go 判定（2026-10-27）

`docs/decision-gate.md` の Gate A「パイロット実行＋baseline実測」のまとめ。
PR #8（データパイロット, merge 2026-10-27T11:52Z）と PR #9（baseline実測,
merge 2026-10-27T12:07Z）を根拠に、案件統括が整理した。

## 1. パイロット実行（データ担当）— 完了

- 権利確認済みの in-repo fixture 1件で、変更前 exit 0 / 変更後 `ZeroDivisionError`
  exit 1 を実測（`docs/pilot-findings-gateA-data.md`）。
- 構造化 1件通過・拒否 0・エラーなし。意味的品質ゲート 1件通過・拒否 0。
- repo別 split: train 1 repo / eval 0 repo、重複なし。
- CI: ruff + mypy / test(py3.11–3.13) / build 全緑。

## 2. baseline実測（独立評価担当→統括代行）— 完了

- `evaluation/run_baseline_eval.py` を Python 3.14.9 venv（macOS gateway,
  CPU）で `sample_cases.json` に対して実行。JSON はハーネス出力の verbatim
  （`evaluation/reports/gateA-baseline-2026-10-27.json`）。
- 実測（n=1、決定論確認済み）:
  - file: Top-1/5/10 = 1.0, MRR = 1.0
  - function: Top-1 = 0.0, Top-5/10 = 1.0, MRR = 0.5
- function-level Top-1 = 0.0 は、lexical baseline が例外フレーム `divide()` を
  root cause `average()` より上位に取る stack-trace-only の期待通りの挙動。
  モデルが埋めるギャップの実測例。
- CI 全緑。
- 注記: 独立評価担当セッションは host=node 固定で gateway 実行不可だったため、
  統括が同条件で代行実行。数値の解釈・制約記載は PR #9 でレビュー済み。

## 3. 制約（Gate B への入力として明示）

1. **n=1**。単一 demo case。性能 claim ではなくハーネスの再現性検証。
   汎用性の効く baseline 数値には権利確認済みの複数 repo が必要。
2. **独立 eval set が作れない**（1 repo のみ）。
3. 単一 repo での train/eval split は不可。
4. BM25 / embeddings 等の比較 baseline（`evaluation/baselines/`）は未実装。
   現 baseline は stack-trace-lexical のみ。
5. `evaluation/run_baseline_eval.py` の `--benchmark` 相対パス指定が cwd 依存で
   失敗する bug あり（絶対パスで回避、別 issue 化推奨）。
6. pytest / フルチェックは実行環境依存（PR #8 の制約記載と同一）。

## 4. 判定

- Gate A の目的（パイロット実行 + 既存能力での baseline 実測 +
  feasibility / 数値 / 条件 / 制約の記録）は**達成**。
- 新規データパイプライン・モデル学習・スケールアップは未着手（Gate A の
  範囲外であること、そのまま維持）。
- 案件統括の推奨: **Go（Gate A 通過）** → Gate B（受入基準の凍結）に進む。

## 5. 次のステップ（Gate B）

`docs/decision-gate.md` の「Acceptance-criteria freeze」テーブルの TBD を、
Gate A の実測と制約を踏まえて**提案値**として埋め、human owner（西山さん）の
承認を得て凍結する。提案前に確認が必要な点:

- 主 KPI（function-level Top-1 / MRR）の閾値と「best baseline 以上の改善幅」
- 評価サンプル数・repo 数（複数 repo 権利確認済みデータが前提）
- 不確実性の扱い（CI / 反復回数）
- レイテンシー・メモリ・モデルサイズの予算

この凍結が得られるまで、trained model の held-out 結果は参照しない
（`docs/evaluation-protocol.md` の分離原則）。
