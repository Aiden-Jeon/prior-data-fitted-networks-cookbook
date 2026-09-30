"""Prior-Data Fitted Networks cookbook의 공용 코드.

각 챕터 노트북에서 반복해서 쓰는 코드(prior, 모델, 학습 루프 등)를 이 패키지에 모아 둔다.
"""

from pfn_cookbook.utils import get_device, set_seed

__all__ = ["get_device", "set_seed"]
