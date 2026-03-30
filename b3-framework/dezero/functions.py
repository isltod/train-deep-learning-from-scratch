import numpy as np
from dezero.core import Function


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        # 이걸 그냥 x로 받으면 튜플 또는 리스트 객체가 담긴다...그 안의 Variable 받으려면 튜플로 받아야된다...
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        # output은 weakref로 연결하고, 약한 참조는 일반 참조와 다르게 ()로 호출해야 한다...
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Sum(Function):
    def forward(self, x):
        y = np.sum(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * np.ones_like(x)
        return gx


def sum(x):
    return Sum()(x)
