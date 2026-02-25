import numpy as np


# 가장 단순한 확률적 경사 하강법...
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # 모멘텀 v도 예상대로 0으로 시작...
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# 이게 모멘텀에서 한 단계 발전시킨 방법이라고...
class Nesterov:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

            for key in params.keys():
                self.v[key] *= self.momentum
                self.v[key] -= self.lr * grads[key]
                params[key] += self.momentum * self.momentum * self.v[key]
                params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            # W와 b 모두 딕셔너리에 넘파이 배열로 처리하는 모양...
            self.h = {}
            for key, val in params.items():
                # 일단 예상대로 h는 0으로 초기화...
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 그냥 * 연산자가 넘파이 원소별 곱
            self.h[key] += grads[key] * grads[key]
            # 그럼, 보정항도 원소별 g / root sum (이전 g^2)이란 얘기...
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            # 또는 같은 식을 이렇게?
            # self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # 그럼 이건 또 뭘까? bias가 있을 때, 그걸 제거하는 버전인가?
            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key])
            # unbias_b += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            # params[key] += lr_t * unbias_m / (np.sqrt(unbias_b) + 1e-7)
