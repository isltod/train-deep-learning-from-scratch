import math
import numpy as np


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        # target은 Model 또는 Layer
        self.target = target
        return self

    def update(self):
        # grad가 None이면 제외하고 업데이트...
        params = [p for p in self.target.params() if p.grad is not None]

        # 매개변수를 전처리하는 훅 함수라고...
        for f in self.hooks:
            f(params)

        # 매개변수 별 업데이트...실제 구현은 상속 클래스에서...
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        # 이게 속도 v인데...일단 아무것도 없이 시작해서...
        self.vs = {}

    def update_one(self, param):
        # 속도 v의 키는 객체 식별자를 이용...
        v_key = id(param)
        # 처음에는 v가 없으니 가중치와 같은 모양의 0 행렬 만들고
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        # 385쪽 식 46.1
        v = self.vs[v_key] * self.momentum - self.lr * param.grad.data
        # 385쪽 식 46.2
        param.data += v


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1.0 - math.pow(self.beta1, self.t)
        fix2 = 1.0 - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (np.sqrt(v) + eps)
