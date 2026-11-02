# FailureLens

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | [한국어](README.ko.md)

**AI를 활용한 오류 원인 위치 분석 및 개발자 도구 모음.**
FailureLens는 오류 로그(스택 트레이스)와 소스 코드 저장소를 바탕으로,
오류의 원인으로 추정되는 코드 위치의 순위를 매기고 각 후보의 **근거**를 제시합니다.
개발자, CI 시스템, AI 코딩 에이전트가 근본 원인을 조사할 수 있도록 돕습니다.

[![CI](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml/badge.svg)](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](docs/installation.md)

이 README는 [영어판](README.md)의 한국어 번역입니다. 상세 문서는 영어로 제공됩니다.

> **현재 상태.** FailureLens는 **실제로 동작하며 의존성이 적은 베이스라인**을 제공합니다
> (스택 트레이스와 어휘 일치 기반, CPU만 사용, 네트워크 불필요).
> **실험 단계의 학습된 인코더**가
> [LevelCapTech/failurelens](https://huggingface.co/LevelCapTech/failurelens)에 공개되어 있습니다.
> 독립적인 홀드아웃 평가는 아직 수행되지 않았으며, 베이스라인이나 기본 인코더보다 우수한지는 확인되지 않았습니다.
> 모델 카드에 있는 SentenceTransformers 직접 사용 예제를 참고하세요.
> 라이브러리의 일반적인 `from_pretrained` 경로는 이 학습된 모델 저장소의
> 사용자 정의 코드와 쿼리 변환을 아직 지원하지 않습니다.

## FailureLens를 사용하는 이유

스택 트레이스는 *오류가 드러난 위치*를 알려 주지만, 항상 *오류의 원인*을 알려 주지는 않습니다.
FailureLens는 두 가지를 구분하여 원인 위치 후보의 순위를 매기고 **실제로 관찰한 근거**만 첨부합니다.
어떤 프레임에 나타났는지, 예외 발생 지점인지, 어떤 오류 토큰이 일치하는지 등을 확인하여 각 제안을 판단할 수 있습니다.
LLM API 래퍼, 범용 코드 검색 도구, 오류 메시지 요약 도구가 아닙니다.

## 기능

- 🔦 **오류 원인 위치 분석 베이스라인** — 투명하고 결정론적인 휴리스틱으로 파일, 함수, 줄 범위의 순위를 매깁니다.
  GPU, 네트워크, 모델 다운로드가 필요하지 않습니다.
- 🧩 **일관된 결과 형식** — Python API와 CLI의 JSON 출력이 버전으로 관리되는 동일한 스키마를 공유합니다.
- 🧪 **결과를 정직하게 표현** — `score_type`은 점수가 확률이 아닌 휴리스틱임을 명시합니다.
  알 수 없는 위치는 추측하지 않고 `null`로 반환하며, 상태 값으로 판단 불가와 처리 실패를 구분합니다.
- 🪶 **가벼운 코어** — 필수 의존성은 `pydantic`과 `typer`뿐입니다. ML 기능은 선택적으로 설치할 수 있습니다.
- 🤖 **에이전트 연동 지원** — 안정적인 종료 코드와 불필요한 출력이 없는 JSON으로 자동화를 지원합니다.
- 📏 **측정 가능** — 재사용 가능한 순위 평가 지표와 실행 가능한 평가 도구를 제공합니다.

## 설치

Python **3.11 이상**이 필요합니다. 아직 PyPI에 배포되지 않았으므로 소스에서 설치하세요.

```bash
git clone https://github.com/LevelCapTech/FailureLens.git
cd FailureLens
pip install -e "packages/python"          # 가벼운 베이스라인만 설치
# pip install -e "packages/python[ml]"    # 선택적 임베딩 백엔드 추가
```

선택적 의존성과 문제 해결 방법은 [설치 가이드](docs/installation.md)를 참고하세요.

## 빠른 시작

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

함께 제공되는 예제에서는 `ZeroDivisionError`가 `divide()`에서 **드러나지만**,
**근본 원인**은 빈 입력을 검사하지 않는 `average()`에 있습니다.
출력된 근거를 통해 각 후보가 선택된 이유를 확인할 수 있습니다.

## 저장소 구조

```text
packages/python/   설치 가능한 라이브러리와 failurelens CLI(제품 본체)
examples/          실행 가능한 데모와 예제 데이터
evaluation/        실측 기반 벤치마크 도구, 평가 지표 문서, 보고서
datasets/          데이터 레코드 스키마와 파이프라인 설계(대용량 데이터나 허구의 데이터는 포함하지 않음)
models/            모델 설계(가중치는 별도 호스팅하며 Git에 저장하지 않음)
training/          실험용 GPU 미세 조정 및 CPU 소규모 데모 파이프라인(선택 실행)
docs/              전체 명세(docs/architecture.md부터 참고)
AGENTS.md          기여자와 AI 에이전트를 위한 공통 작업 규약
```

## 문서

다음 상세 문서는 영어로 제공됩니다.

| 주제 | 문서 |
|------|------|
| 동작 원리 | [docs/architecture.md](docs/architecture.md) |
| 설치 | [docs/installation.md](docs/installation.md) |
| 처음 실행하기 | [docs/getting-started.md](docs/getting-started.md) |
| Python API와 스키마 | [docs/api.md](docs/api.md) |
| CLI와 종료 코드 | [docs/cli.md](docs/cli.md) |
| 데이터셋 | [docs/datasets.md](docs/datasets.md) |
| 학습 | [docs/training.md](docs/training.md) |
| 평가 | [docs/evaluation.md](docs/evaluation.md) |
| 보안 | [docs/security.md](docs/security.md) · [SECURITY.md](SECURITY.md) |
| 모델 카드 | [docs/model-card.md](docs/model-card.md) |
| 로드맵 | [docs/roadmap.md](docs/roadmap.md) |
| 요구사항 | [docs/requirements.md](docs/requirements.md) |
| AI 모델 설계 의사결정 게이트 | [docs/decision-gate.md](docs/decision-gate.md) · [데이터 계약](docs/data-contract.md) · [평가 프로토콜](docs/evaluation-protocol.md) |
| 기여 안내 | [CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) |

## 개발

```bash
make install     # 개발 도구를 포함하여 편집 가능한 모드로 설치
make check       # ruff 검사 및 포맷 검사, mypy 엄격 타입 검사, pytest(CI와 동일)
make demo        # 함께 제공되는 예제에서 베이스라인 실행
make build       # sdist와 wheel 빌드
python evaluation/run_baseline_eval.py   # 베이스라인 평가 지표 실측
```

## 기여

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md)와
[AGENTS.md](AGENTS.md)(작업 규약, 특히 사실에 충실하게 보고하기 위한 규칙)를 읽고,
[행동 강령](CODE_OF_CONDUCT.md)을 따라 주세요.

## 라이선스

이 저장소의 소스 코드에는 Apache-2.0이 적용됩니다. [LICENSE](LICENSE)를 참고하세요.
공개된 실험 모델의 가중치는 MIT 라이선스를 따르며, 업스트림 코드의 권리 고지를 유지합니다.
자세한 내용은 [모델 카드](docs/model-card.md)를 참고하세요.
