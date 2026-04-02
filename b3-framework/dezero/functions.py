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


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None
        # 일반적으로 b가 있다면 그건 분기해서 사용하므로 여기서는 원래 b 모양대로 합산
        if b.data is not None:
            gb = sum_to(gy, b.shape)
        # 336쪽 식 41.3, 41.4
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


# 이건 dezero 클래스 안 쓰고 사칙연산 이용하는 버전이라,
# backward 안만들어도 이미 만들어진 사칙연산에서 역전파 된다...
def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    # 요걸로 메모리 효율을 유지한다고...
    t.data = None
    return y


# e^x는 순전파도 역전파도 e^x
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        # output은 weakref, 일반 참조와 다르게 ()로 호출...
        y = self.outputs[0]()
        # output이 e^x
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        # sigmoid 미분
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


# 이것도 만들어둔 사칙연산 써서 하는 방법...
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        # 그냥 보통의 넘파이 슬라이싱, 이 부분만 선택한다는 건데...
        # 결국 x를 x0, x1으로 쪼개고(역전파는 더하기), 각각에 0과 1을 곱한다(역전파도 0과 1 곱하기)는 얘기...
        y = x[self.slices]
        return y

    def backward(self, gy):
        (x,) = self.inputs
        # 역전파 미분은 그 선택 슬라이스와 원래 입력의 모양을 가지고...
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        # 원래 모양의 0 행렬에, 선택했던 부분만 위에서 받은 미분을 넣어준다...
        # 1 곱했던 부분에 1 곱해주고, 그걸 0 곱했던 부분과 더한다 정도...
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        # 여긴 forward니까 x는 넘파이...
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        # 약한 참조니까 ()로 출력 변수 받고
        y = self.outputs[0]()
        # Softmax 미분 수식과 코드 참고...
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        # x의 값들을 min~max 사이로 맞추기
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        # 그럼 이 경우 미분은 min max 사이의 유효 x에만 곱하기 1로 봐야 하나? 나머진 0?
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class Log(Function):
    def forward(self, x):
        # forward는 넘파이 로그
        y = np.log(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        # backward는 dezero의 나누기로 1/x
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


def softmax_cross_entropy_simple(x, t):
    # x는 벡터, t는 스칼라로 인덱스?
    x, t = as_variable(x), as_variable(t)
    # N은 배치 수?
    N = x.shape[0]
    p = softmax(x)
    # log(0)을 방지한다...그냥 min으로 하면 안되나?
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    # log_p에서 배치 수대로 돌면서 t 인덱스의 정답만 추출...
    tlog_p = log_p[np.arange(N), t.data]
    # 평균 손실 반환
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    # x는 모델 예측 결과 y(이게 클래스 별 점수), t는 정답 위치 스칼라
    def forward(self, x, t):
        # 이건 배치 갯수...
        N = x.shape[0]
        # softmax cross entropy가 생각해보면...
        # log(exp(xi) / sum(exp(x))) = log(exp(xi)) - log(sum(exp(x))) = xi - log(exp(x1) + exp(x2) + ...)
        # 따라서 뒤 log sum exp 부분을 먼저 구하고
        log_z = utils.logsumexp(x, axis=1)
        # 그걸 x에서 빼면 SCE...
        log_p = x - log_z
        # 이건 배치 별로 정답지 뽑는거고...
        log_p = log_p[np.arange(N), t.data]
        y = -np.sum(log_p) / N
        return y

    def backward(self, gy):
        (x, t) = self.inputs
        # 원래 입력 x(사실 y)가 배치 * 클래스
        N, CLS_NUM = x.shape
        # 이건 평균 구한 부분 보충, 결국 1/N 곱하기였으니까...
        gy *= 1 / N
        y = softmax(x)
        # 여긴 선택 없이 그냥 쿠파이 쓰라는데...문제가 좀 있네 코드가...
        # 암튼 정답지 원핫 인코딩 만드는 코드인 듯...
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        # SCE 미분은 y-t...해당 위치에서...
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
