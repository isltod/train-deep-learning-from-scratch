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
