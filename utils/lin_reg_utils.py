from typing import Tuple, List
import numpy as np


def get_data(slope: float, bias: float):
    n: int = 100
    x = np.linspace(-1, 1, n)
    y = slope * x + np.random.normal(0, 0.5, size=x.shape) + bias

    return x, y


def train_params(x, y, w_init: float, b_init: float, lr: float, steps: int,
                 no_bias: bool) -> Tuple[List[float], List[float]]:
    n: int = x.shape[0]
    w: float = w_init
    b: float = b_init
    w_l: List[float] = [w]
    if no_bias:
        b = b * 0
    b_l: List[float] = [b]
    for i in range(steps):
        w_grad = -np.sum((y - (w * x + b)) * x)/n
        b_grad = -np.sum((y - (w * x + b)))/n
        w = w - lr * w_grad
        b = b - lr * b_grad

        w_l.append(w)
        if no_bias:
            b = b * 0
        b_l.append(b)

    return w_l, b_l
