import numpy as np
from dezero.core import Function, as_variable


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


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        # 여기 x는 스칼라 또는 ndarray이고...numpy 메서드로 변환
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        # 여기 gy는 Variable 클래스라서 numpy아닌 dezero 함수로 변환
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        # 순전파 목표 축 정보 받아두고
        self.axes = axes

    def forward(self, x):
        # 여기서도 x는 ndarray이므로 np의 transpose
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        # 원래 축 변환에 별다른 것이 없다면 표준 transpose 반환
        if self.axes is None:
            return transpose(gy)

        # 아니고 뭔가 다르게 변환하도록 축 정보가 있었다면 그 역순으로 정리...
        axes_len = len(self.axes)
        # argsort - 정렬할 때 인덱스, % - 나머지...근데 오른쪽은 이러면 self.axes와 같은 결과 아닌가?
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        # 하지만 gy는 Variable이므로 dezero의 transpose를 사용해야 한다...
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)
