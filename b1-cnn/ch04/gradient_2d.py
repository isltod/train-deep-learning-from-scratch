import numpy as np
import matplotlib.pylab as plt


# 이건 여태껏 해오던 함수가 아닌데...모든 값을 다 제곱해서 더하는 함수?
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    # 순서쌍에서 한쪽 축의 값들을 배열로 받아서, 같은 크기 같은 형식의 0 배열 만들고
    grad = np.zeros_like(x)
    for idx in range(x.size):
        # 각 값마다 돌면서
        tmp_val = x[idx]
        # 나머진 그대로, 현재 좌표에 대해서만 h 더하고 증가 함수 구하고
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        # 수치미분이라 원래 함수가 아니라 감소 함수로 계산...
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 현재 좌표값 원상복구 - 좌표값 배열을 다음에 또 써야 하니까
        x[idx] = tmp_val
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        # 받은 좌표쌍(2, 324)과 같은 형태, 같은 타입의 0 배열
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            # 0 줄 배열(324), 1줄 배열(324) 순으로 반복
            grad[idx] = _numerical_gradient_no_batch(f, x)
        return grad


if __name__ == "__main__":
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    # 메트릭스 아니고, X와 Y가 x0와 x1의 가능한 모든 좌표쌍을 이루는 개별 벡터가 된다...
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()
    # f2는 x^2의 2차원 함수, -2.5~2.5 사이의 각 좌표쌍에서 편미분...
    grad = numerical_gradient(function_2, np.array([X, Y]))
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
