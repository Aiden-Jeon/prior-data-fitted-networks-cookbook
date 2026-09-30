# Prior-Data Fitted Networks Cookbook

Prior-Data Fitted Networks(PFN)를 밑바닥부터 한 단계씩 구현하면서 이해하는 cookbook입니다.

PFN은 prior에서 뽑은 합성 데이터셋을 대량으로 학습해, 새 데이터셋이 들어오면 forward pass 한 번으로
posterior predictive distribution(PPD)을 근사하는 모델입니다. TabPFN도 이 아이디어에서 나왔습니다.

## 로드맵

| # | 챕터 | 핵심 질문 |
|---|------|-----------|
| 00 | 환경 점검 | 실습 환경이 제대로 준비됐는가? |
| 01 | 베이지안 추론과 PPD | PFN이 근사하려는 대상 $p(y \mid x, D)$는 무엇인가? |
| 02 | Prior에서 데이터셋 샘플링 | GP prior, BNN prior로 합성 데이터셋을 어떻게 만드는가? |
| 03 | Prior-Data NLL | 합성 데이터에서 NLL을 최소화하면 왜 PPD 근사가 되는가? |
| 04 | 순열 불변 Transformer | context/query를 어떻게 넣고, attention mask는 왜 필요한가? |
| 05 | Riemann(Bar) Distribution | 회귀 출력을 왜 구간 분류로 바꾸는가? |
| 06 | 첫 PFN: 1D GP 회귀 | 직접 학습한 PFN이 정확한 GP posterior와 얼마나 비슷한가? |
| 07 | 분류와 SCM prior | 표 형식 데이터를 위한 prior는 어떻게 설계하는가? |
| 08 | TabPFN 살펴보기 | 실제 TabPFN은 무엇이 다르고 어떻게 쓰는가? |
| 09 | 확장 주제 | 베이지안 최적화, 학습 곡선 외삽 등으로 어떻게 넓히는가? |

## 프로젝트 구조

```
.
├── notebooks/            # 챕터별 노트북 (00_..., 01_..., ...)
├── src/pfn_cookbook/     # 노트북에서 재사용하는 코드 (prior, 모델, 학습 루프)
├── tests/                # src 코드 테스트
└── pyproject.toml
```

노트북에서 처음 구현한 코드가 다음 챕터에서도 필요해지면 `src/pfn_cookbook/`로 옮겨서 `import`해 씁니다.

## 시작하기

[uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync                 # 가상환경 생성 + 의존성 설치
uv run jupyter lab      # 노트북 실행
```

그 밖의 명령:

```bash
uv run pytest           # 테스트
uv run ruff check .     # 린트
uv run ruff format .    # 포맷
```

노트북 출력은 `nbstripout` git filter가 커밋할 때 자동으로 지웁니다. 새로 clone했다면 한 번 설치하세요.

```bash
uv run nbstripout --install --attributes .gitattributes
```

## Databricks Serverless GPU에서 실행하기

PFN 학습처럼 무거운 연산은 Databricks Serverless GPU에서 돌립니다. 로컬(Apple Silicon)은 코드 작성과
작은 실험용입니다. 같은 노트북을 두 곳에서 그대로 실행할 수 있게 맞춰 두었습니다.

1. Workspace에서 이 저장소를 **Git folder**로 clone합니다.
2. 노트북을 열고 오른쪽 **Environment** 패널에서 accelerator(A10 또는 H100)와 환경 버전을 고릅니다.
   `pyproject.toml`의 의존성 하한은 Serverless GPU 환경 v6(Python 3.12, torch 2.11 + CUDA 13.0)에 맞춰져 있습니다.
3. `notebooks/00_setup_check.ipynb`을 실행해 `device: cuda`와 GPU 이름이 나오는지 확인합니다.

각 노트북의 첫 셀은 `pfn_cookbook`이 설치되어 있지 않으면 저장소의 `src/`를 import 경로에 추가합니다.
그래서 Databricks에서 따로 패키지를 설치하지 않아도 공용 코드를 쓸 수 있습니다.

## 참고 자료

- Müller et al., *Transformers Can Do Bayesian Inference*, ICLR 2022 — PFN 원 논문
- Hollmann et al., *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*, ICLR 2023
- Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 2025 — TabPFN v2
- [automl/PFNs](https://github.com/automl/PFNs) — 공식 PFN 구현
- [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
