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
├── src/pfn_cookbook/     # 노트북에서 재사용하는 코드 (prior, 모델, 학습 루프, 검증)
├── scripts/              # GPU에서 돌리는 실행 스크립트 (학습, 환경 점검)
├── air/                  # AI Runtime CLI workload 설정
├── tests/                # src·scripts 테스트
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

Databricks CLI는 `e2-demo-field-eng` 프로필을 씁니다.

```bash
databricks auth login --profile e2-demo-field-eng   # 토큰이 만료됐을 때
databricks current-user me --profile e2-demo-field-eng
```

환경은 워크스페이스 base environment **AI v6**(`databricks_ai_v6`: Python 3.12, torch 2.11.0+cu130)를 씁니다.
`pyproject.toml`의 torch 고정 범위와 의존성 하한도 이 환경에 맞춰져 있습니다.

실행 방법은 세 가지입니다.

### 1. AI Runtime CLI(`databricks air`): 학습 스크립트 실행

PFN 학습처럼 끝까지 돌려야 하는 작업에 씁니다. [AI Runtime CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart)가 로컬의 `src/`·`scripts/`를 스냅샷으로 올려 Serverless GPU에서 `scripts/*.py`를 실행합니다. 커밋하지 않은 변경도 그대로 올라갑니다. workload 설정은 `air/`에 있습니다. CLI v1.19 이상이 필요합니다(`brew upgrade databricks`).

```bash
databricks air run -f air/setup_check.yaml --watch -p e2-demo-field-eng   # A10에서 환경 점검
databricks air run -f air/train_pfn.yaml --watch -p e2-demo-field-eng     # 학습 (large, 500 step)
```

규모는 yaml을 고치지 않고 `--override`로 바꿉니다. 논문 재현(A10 기준 2~3시간)은 다음과 같습니다.

```bash
databricks air run -f air/train_pfn.yaml --watch -p e2-demo-field-eng \
  --override env_variables.PFN_CONFIG_NAME=paper --override env_variables.PFN_STEPS=10000 \
  --override mlflow_run_name=paper-10000steps --override timeout_minutes=240
```

제출하면 Job Run ID가 나옵니다. `databricks air get <id>`로 상태와 Job·MLflow run 링크를, `databricks air logs <id>`로 로그를 봅니다. MLflow run(experiment `/Workspace/Users/jongseob.jeon@databricks.com/pfn-cookbook/pfn-gp-reproduction`)에는 config, `train_nll`/`val_nll`, analytic GP 대비 지표(`rmse_mean`, `nll_gap` 등), loss curve·오버레이 그림, best/final 모델이 남습니다. 모델은 `load_logged_model("runs:/<run_id>/best_model")`로 불러옵니다.

같은 스크립트를 로컬에서도 돌릴 수 있습니다: `uv run python scripts/train_pfn.py --config-name small --steps 50`

### 2. Asset Bundle: 노트북을 job으로 실행

노트북을 그대로 job으로 돌릴 때 씁니다. job 정의는 `databricks.yml`에 있습니다.

```bash
databricks bundle deploy                 # 로컬 코드를 워크스페이스에 업로드 + job 생성/갱신
databricks bundle run setup_check        # 00 노트북을 A10 GPU에서 실행
```

Serverless notebook task에 GPU를 붙이려면 task에 `compute.hardware_accelerator`(`GPU_1xA10` 또는
`GPU_8xH100`)와 `environment_key`를 함께 지정해야 합니다.

### 3. Git folder: 워크스페이스에서 노트북을 직접 실행

읽으면서 셀 단위로 실험할 때 씁니다. Git folder 경로는
`/Workspace/Users/jongseob.jeon@databricks.com/prior-data-fitted-networks-cookbook`입니다.

1. 로컬에서 push한 뒤 Git folder에서 **Pull**하거나 CLI로 갱신합니다.
   ```bash
   databricks repos update /Workspace/Users/jongseob.jeon@databricks.com/prior-data-fitted-networks-cookbook \
     --branch main --profile e2-demo-field-eng
   ```
2. 노트북을 열고 오른쪽 **Environment** 패널에서 accelerator(A10 또는 H100)와 base environment **AI v6**를 고릅니다.
3. `notebooks/00_setup_check.ipynb`을 실행해 `device: cuda`와 GPU 이름이 나오는지 확인합니다.

각 노트북의 첫 셀은 `pfn_cookbook`이 설치되어 있지 않으면 저장소의 `src/`를 import 경로에 추가합니다.
그래서 Databricks에서 따로 패키지를 설치하지 않아도 공용 코드를 쓸 수 있습니다.

## 참고 자료

- Müller et al., *[Transformers Can Do Bayesian Inference](https://arxiv.org/pdf/2112.10510)*, ICLR 2022 — PFN 원 논문
- Hollmann et al., *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*, ICLR 2023
- Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 2025 — TabPFN v2
- [automl/PFNs](https://github.com/automl/PFNs) — 공식 PFN 구현
- [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
