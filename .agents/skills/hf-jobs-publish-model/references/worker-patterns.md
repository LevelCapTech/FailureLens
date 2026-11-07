# Cloud workerの実装パターン

既存workerがある場合は、そのプロジェクト規約と実際のimageを優先する。APIや単価は実行時のSDKと一次資料で確認する。

## モデルの同一性と来歴

Remote artifactの再パッケージ化なら元digestを照合する。再学習なら独自digestを保存し、別GPUや異なるライブラリで生成したモデルに、既存artifactのhash・精度・評価済み状態を引き継がない。

記録する情報はsource commit/archive digest、image、base revision、データdigestと利用条件、秘密の本文を含まない件数・split、実効設定、seed、依存version、検出したGPU、生成物のサイズ・SHA-256、Job ID、公開commit、ロード検証結果。

設定値が実装に適用されているか確かめる。例えば`max_length`が学習関数で使われていなければ、設定上の値を実効値として記載しない。

## OS管理Pythonと依存関係

まずimage内の既存venvとPythonを調べる。適切な環境があるならそれを使う。

PyTorchとpipがsystem siteにあり、PEP 668で追加インストールが拒否されるimageでは、次のパターンが使える。

```python
import os
import tempfile
import venv

root = tempfile.mkdtemp(prefix="model-build-venv-")
venv.EnvBuilder(system_site_packages=True, with_pip=False).create(root)
python = os.path.join(root, "bin", "python")
# このpythonで必要な依存を入れ、同じpythonでworkerを実行する。
```

これは既存pip/PyTorchを再利用する条件付きの方法で、あらゆるimageの既定解ではない。pipやvenvがないimageでは適切なimageを選び、system-site継承で使う依存の互換性を確認する。`--break-system-packages`やセキュリティ下限の引き下げを通常の対処にしない。

`training.finetune`のimportでも`training/__init__.py`が先に動く。呼び出す関数だけから依存を判断せず、パッケージ初期化まで確認する。

## 小容量の非公開入力

検証済みloaderから必要な特徴量だけを作り、秘密や私的なパスを除去する。圧縮は暗号化や公開許可の代わりではない。送信サイズとAPI制限を確認し、承認された非公開転送にのみ使う。

```python
from huggingface_hub import HfApi, get_token

api = HfApi()
# 承認済みの転送・費用・公開範囲から変数を決めた後に実行する設計例。
job = api.run_job(
    image=reviewed_image,
    command=worker_command,
    secrets={"HF_TOKEN": get_token(), "PRIVATE_INPUT": private_payload},
    env=public_build_settings,
    flavor=selected_hardware,
    timeout=remaining_timeout,
    attempts=1,
    namespace=approved_job_namespace,
)
save_public_job_state({"id": job.id, "url": job.url})
```

設計例の変数は現在の案件から定める。トークンやpayloadを表示しない。Job ownerとmodel repo ownerは異なる場合がある。組織repoだからといってJobも同じnamespaceにあると推測しない。

`run_uv_job`でローカルscriptを渡す方式は、実装によってscript用の一時repoを作る。入力の転送経路を調べ、秘密の本文をscriptや一時repoへ埋め込まない。

## カードを先に検証する

アップロード関数の最後の検証だけに頼ると、学習後にカード不備が判明する。

```python
from pathlib import Path
from huggingface_hub import ModelCard

ModelCard(Path("README.md").read_text(encoding="utf-8")).validate(repo_type="model")
```

`language: [en]`等の言語コードを使い、Python対応は`tags: [python]`や本文へ書く。license、base model、library、pipelineは実体と一致させ、仮の評価値を公開しない。

## 再試行と総予算

観測失敗では同じIDを再照会する。`rerun_job`は別IDの有料Jobを作るため、状況確認に使わない。

単価が分か秒かを確かめ、総予算から既に使った分と余裕を差し引く。同じhardwareの非並列Jobなら、初回作成から再試行までの全経過時間を使用済みとする方法は保守的な上限になり得る。異なるhardwareや並列Jobでは各単価と利用量から積算する。

再試行前に実終了、原因に対応した修正、既にHubへ書き込まれた成果物、残予算を確認する。公開後の検証だけが失敗した場合は、まず既存commitを検証し、再学習を自動で始めない。

## FailureLensの場合

checkout内の`docs/hf-jobs-publication.md`と`scripts/hf_jobs_release.py`を確認する。既存scriptは特定データdigest、base revisionと公開先に結び付いており、別モデル向けの汎用launcherではない。過去の公開承認・価格・Job IDを新しい公開の権限や現在の価格にしない。

## 一次資料

- [Hugging Face Jobs SDK](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
- [Model cards and validation](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards)
- [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)
- [Hub authentication](https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication)
