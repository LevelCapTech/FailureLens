---
name: hf-jobs-publish-model
description: Build and publish model weights with Hugging Face Jobs while keeping large checkpoints off the submitting connection. Use for Jobs-based model releases, republishing, or troubleshooting an existing publication Job, including local upload restrictions.
---

# Hugging Face Jobsでモデルを公開する

モデルの取得・必要な学習・変換・重みのアップロードをHugging Face内で実行し、公開された固定revisionを実際に読み直してから完了を報告する。

## 着手と送信範囲

- プロジェクト規約、公開先repo、現在のHubファイル、既存Job IDと実状態を確認する。指定されたモデル・アカウント・配布形式を維持する。
- 目的の重みが既にcloudにあるなら取得してパッケージ化する。手元の重みを送れず再学習が必要なら、元の条件から再構築できるか確かめる。再構築した重みは別成果物としてhashを記録し、元artifactと同一とは主張しない。未学習base modelで代用しない。
- ローカル送信制約を守り、スクリプトや必要な小容量入力だけを送る。大きな重み・データ・中間生成物は承認されたcloud側で取得・生成する。送信先・用途・サイズ・公開範囲と費用上限を具体化する。
- 同じセッション内の既存承認を引き継ぎ、同じ内容を再確認しない。過去の別案件の承認を新しいデータ・宛先・費用へ拡張しない。不足する承認がある場合も、worker・カード・転送内容・予算を先にレビュー可能な状態にする。スキル作成や呼び出し自体は、有料Jobや新しい非公開データ転送の承認ではない。

## ビルド・公開

1. CLI/SDKの認証状態とnamespaceへの権限を確認する。既存の`hf auth login`等を使い、トークン値を表示したりチャットへ貼らせたりしない。
2. 既存のproject固有workerを優先する。source commit、base revision、image、依存version、入力digest、実際の学習条件と出力形式を記録する。カスタムコードは実行する固定revisionを確認し、プロジェクトのセキュリティ下限を下げない。
3. 有料Job起動前に`ModelCard(...).validate(repo_type="model")`で公開予定カードをHub検証する。`language`は`en`等の言語コードにし、`python`等はタグへ置く。学習パッケージの`__init__`が読み込む依存も調べる。
4. cloud imageで使うPythonを確認する。OS管理Pythonへの強制インストールを前提にせず、既存環境か専用venvを使う。GPU/CUDA、必要なimport、保存形式を学習より先に確認する。workerの実装・修正時は[worker-patterns.md](references/worker-patterns.md)を読む。
5. 秘密値や非公開の小容量入力はJobsの`secrets`で渡す。引数・通常の`env`・コード・ログ・公開ファイルに埋め込まず、Jobオブジェクト全体も表示しない。公開物は明示した出力先またはallowlistから選び、生データ・optimizer・認証情報を混ぜない。
6. 現行単価と合意済み総予算からhardware、timeout、試行数を決める。最初は自動再試行なしを基本にし、別設定なら全試行を予算へ計上する。作成直後にJob ID、URL、namespace、制限とsource hashを秘密値なしで保存する。
7. 同じJobを監視する。観測timeoutやログ取得失敗だけで再起動しない。`RUNNING`や`SCHEDULING`なら同じIDを再照会する。実際の`ERROR`等の終了を確認して原因を直し、残予算、既存成果物、公開先の親commitを確認してから再試行する。同じ原因を未修正のまま繰り返さず、残予算や権限が不足したら追加起動を止める。
8. worker内で出力のロード、入出力、非有限値等を確認してからアップロードする。親commitを指定する等、並行更新の上書きを防ぐ。公開後は返された不変commitと別cacheでロードし、出力とhashを照合する。カードのみ修正した場合も、使用例のrevisionが同じ重みを参照することを確かめる。

## 完了条件

- 最終の対象Jobが`COMPLETED`で、目的の成果物が公開されている。複数の失敗通知はJob IDごとに最終Jobと区別する。
- 公開repoは認証なしで確認する。privateやgatedを指定された場合は、そのアクセス条件どおりに確認する。
- 重み、tokenizer、config、必要な推論コード、カード、ライセンスと来歴が揃い、サイズとhashが生成時記録と一致する。巨大ファイルはcloudで完全検証し、ローカルではHubのhashと小さなmanifest等を照合する。
- 公開された固定revisionを実際にロードして推論できる。汎用helperやファイル存在確認だけで満たしたことにしない。
- 説明と使用例が実装に合っている。独立評価の未完了、対応言語、候補取得との境界、未校正スコア、API統合の制約を明記し、ロード成功を精度や優位性の証明にしない。
- 不要な有料Jobが稼働し続けていないことを確認し、URL、revision、検証結果と残る制約を簡潔に報告する。実費は請求記録なしで断定しない。

## この環境(大容量アップロード禁止の回線)での使い方

- 回線が大容量アップロードを禁じているため、重み・データ・チェックポイントをローカルから`upload_file`/`hf upload`等で送らない。Hubへの登録は`huggingface_hub`のJobs API(`hf jobs`)で、worker側がHub内で取得・生成・アップロードする。ローカルから送るのはworkerスクリプトと小容量の入力だけ。
- Claude Codeからも`.claude/skills/`経由で使える。`huggingface_hub`はプロジェクトの`.venv`には入っていない。`hf` CLIの専用環境のPythonを使う: `HFPY=$(head -1 ~/.local/bin/hf | sed 's/^#!//')`。そのうえで`$HFPY <skill-dir>/scripts/inspect_release.py ...`や`hf jobs ...`を実行する。`.venv`へ追加インストールしない(コア依存を増やさない)。
- 認証は既存の`hf auth`のログイン(namespace `LevelCapTech`)を使う。トークン値は表示しない。
- FailureLensの規約を優先する: 公開、学習の起動、有料Job、新しい非公開データの転送は、人間オーナーの明示承認がある場合だけ実行する。このスキルの呼び出しは承認ではない。承認がなければ、worker・カード・予算案を整えてレビューを求めるところで止める。読み取り専用の確認(`inspect_release.py`、`hf jobs list/inspect`)は承認なしで行える。
- 公開後は、MLflowへの記録が必要なら`track-experiments-mlflow`スキルに従い、`failurelens_provenance.json`を`import_history`で取り込む。ロード成功を精度や優位性の根拠にしない。

## 読み取り専用helper

`huggingface_hub`が入ったPythonで、`scripts/inspect_release.py`を使える。

```bash
python3 <skill-dir>/scripts/inspect_release.py \
  --repo owner/model --public \
  --job https://huggingface.co/jobs/owner/job-id
```

`--job`は複数指定できる。裸のIDには`--job-owner`が必要。`--hardware t4-small`で現在の単価、`--card /path/to/README.md`でカードのHub検証を行う。カード検証は本文をHubへ送るため、秘密値のない公開予定カードだけを渡す。

helperはJob作成・再試行・repo変更・ログ表示・重み本体の取得をしない。結果は状態とメタデータであり、推論検証や公開成功の総合判定ではない。
