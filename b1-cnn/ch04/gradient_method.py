import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    for i in range(step_num):
        # 히스토리에 현재 위치 저장하고
        x_history.append(x.copy())
        # 경사하강법
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


if __name__ == "__main__":
    # 처음에 (-3, 4) 지점에서 출발
    init_x = np.array([-3.0, 4.0])
    # 적정 학습률
    # lr = 0.1
    # 너무 큰 학습률
    # lr = 10.0
    # 너무 작은 학습률
    lr = 1e-10
    step_num = 20
    # 함수 f2 구배를 따라 경사하강법... 단계별 x의 이동은 history에 (총 스텝, 2) 배열로
    x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
    print(x)
    print(x_history)
    plt.plot([-5, 5], [0, 0], "--b")
    plt.plot([0, 0], [-5, 5], "--b")
    plt.plot(x_history[:, 0], x_history[:, 1], "o")
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
