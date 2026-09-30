import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Python, NumPy, PyTorch의 난수 시드를 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """사용 가능한 가장 빠른 장치를 고른다 (cuda > mps > cpu).

    `prefer`를 지정하면 그 장치를 그대로 쓴다. 예: `get_device("cpu")`.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
