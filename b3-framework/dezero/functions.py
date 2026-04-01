import numpy as np
from dezero.core import Function, as_variable
from dezero import utils


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


# 이 밑으로는 Sum 클래스 하나 도입하는데, 설명없이 막 갖다 써서 붙여놓은 것들인데...
class BroadcastTo(Function):
    def __init__(self, shape):
        # 목표 shape...이걸 왜 forward에서 안받고 생성하면서 받을까?
        # Function.__call()__ 에서 foward 호출할 때 입력 x만 넘기게 해서 인수 2 이상이면 생성자 필요한가?
        self.shape = shape

    def forward(self, x):
        # 입력 변수 shape 저장하고, 목표 shape로 확장 공사해서 반환...numpy 버전
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        # 반대로 원래 shape로 합산해서 반환...dezero 버전...
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        # 이것도 목표 shape
        self.shape = shape

    def forward(self, x):
        # 입력 ndarray shape 받아서 저장해두고, 차원 고려해서 합산 반환...dezero 버전...
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        # 위에서 내려온 미분 gy를 원래 형상으로 확장 공사해서 반환
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        # 일단 이건 np.sum()일테고...
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        # axis와 keepdims 때문에 뭔가 변할 수 있어서 미세하게 형상 조정하는 함수?
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        # 입력변수와 shape가 같아지도록 부풀려서 반환
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class Matmul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        # 여기 전치는 넘파이 아니고 Variable에 만든 transpose
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return Matmul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        # sum()은 gy를 gx 원소들 각각에 곱하는 걸로(분기, 브로드캐스트) 돌리고,
        # N = len(diff)는 상수니 그대로, diff^2은 2diff로...
        gx0 = gy * diff * (2.0 / len(diff))
        # 뒤 변수는 앞과 같지만 - 기호만 추가...
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)
