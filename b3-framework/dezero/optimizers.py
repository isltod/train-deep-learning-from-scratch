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
        v_key = id(param)
        # 처음에는 v가 없으니 가중치와 같은 모양의 0 행렬 만들고
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        # 385쪽 식 46.1
        v = self.vs[v_key] * self.momentum - self.lr * param.grad.data
        # 385쪽 식 46.2
        param.data += v
