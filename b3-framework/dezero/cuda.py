gpu_enable = True
try:
    import cupy as cp
    import cupyx as cpx

    cupy = cp
    # scatter_add 대응...
    cupyx = cpx
except ImportError:
    gpu_enable = False
import numpy as np
from dezero import Variable


def get_array_module(x):
    # Variable, 넘파이 배열, 쿠파이 배열 중에 하나를 받나? 일단 Variable이면 넘파이 또는 쿠파이 배열로?
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np

    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception("CuPy cannot be loaded. Install CuPy!")

    return cp.asarray(x)
