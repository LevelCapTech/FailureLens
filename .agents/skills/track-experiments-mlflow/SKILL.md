---
name: track-experiments-mlflow
description: Record FailureLens evaluation and training results in the local MLflow store and check what is already registered. Use after any baseline evaluation, toy or Stage 2 training run, or when importing earlier training records, and before comparing runs or reporting experiment history.
---

# MLflowへ実験結果を登録する

評価・学習を実行したら、その実測結果を `failurelens-development` 実験へ記録する。正本は `docs/experiment-tracking.md`。ここでは手順と判断だけを示す。

## 1. 登録状況を確認する(最初に必ず)

```bash
source .venv/bin/activate   # 未導入なら make tracking-install
python .agents/skills/track-experiments-mlflow/scripts/mlflow_status.py
```

読み取り専用。`failurelens.kind`(`baseline-evaluation` / `toy-training` / `stage2-training`)ごとに実行を一覧し、0件の種別を示す。`failurelens.git_dirty=true`の実行は、未コミット変更を含んだ状態の記録であり、比較の根拠にする前に注意する。

## 2. 種別ごとの記録方法

| 実行 | コマンド | kind |
| --- | --- | --- |
| baseline評価 | `python evaluation/run_baseline_eval.py --mlflow --mlflow-run-name <名前>` (または `make eval-track`) | `baseline-evaluation` |
| CPU toy学習 | `python -m training.scripts.smoke_train --mlflow --mlflow-run-name <名前>` | `toy-training` |
| Stage 2学習 | `python -m training.scripts.finetune_train --config ... --mlflow --mlflow-run-name <名前>` | `stage2-training` |
| 過去の学習記録 | `python -m training.scripts.import_history <experiment_metadata.json または failurelens_provenance.json> --mlflow --mlflow-run-name <名前>` | `stage2-training` + `failurelens.historical=true` |

- `--mlflow` なしでは記録されない。環境変数だけでは有効にならない。
- run名は内容が分かる名前にする(例: `after-<変更内容>`)。コード変更の前後比較は同じbenchmark・candidate limit・環境で2回記録し、UIの **Compare** で見る。
- 再開学習(`--resume-from`)は新しいrunになる。過去のlossを新規測定として再記録しない。
- 記録の前にコミットしておくと `failurelens.git_dirty=false` になり再現しやすい。コミットは依頼があったときだけ行う。

## 3. 記録後の確認

1. 手順1のスクリプトを再実行し、新しいrunが `FINISHED` で、期待するmetricが入っていることを確認する。`FAILED` なら原因を直して再実行する(失敗runを成功扱いにしない)。
2. 必要なら `make mlflow-ui` (http://127.0.0.1:5000) で比較する。
3. 数値を報告するときは、そのrunのmetricをそのまま使い、分母(`num_cases`)とbenchmarkのdigestを併記する。

## 守ること

- 同梱benchmarkは1件のデモ。回帰確認には使えるが、一般化の根拠や優位性の主張には使わない。
- 学習のloss(toyを含む)は精度ではない。importした損失曲線も品質評価ではない。
- 独立hold-out評価(Gate C)の結果は、この経路で自動登録しない。登録するかどうか、集約値のみにするかは、人間オーナーの承認を取ってから決める。ケース別出力、ラベル、生トレースは記録しない。
- 重み、チェックポイント、データ本体、秘密情報をMLflowへ入れない。共有サーバーを使う場合は明示的に指定し、認証情報をコマンドや設定へ書かない。
- `.mlflow/` はGit管理外。バックアップは書き込みとUIを止めてディレクトリごと行う。
- 学習・公開・データ取得の実行そのものは、このスキルでは承認されない。
- 存在しない数値を補わない。記録がなければ「未登録」と報告する。

## 4. 結果の報告

登録状況(種別ごとの件数、今回追加したrun名とID)、スキップや失敗、未登録の項目を簡潔に伝える。未登録の主な項目の例: 実データでの複数ケース評価、Gate C。
