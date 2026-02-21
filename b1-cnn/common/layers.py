import numpy as np
from common.functions import softmax, cross_entropy_error


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        # 결국 Relu는 0 이하는 0, 0 이상은 그대로(y=x)라는 로직
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 미분이 0 이하는 0, 이상은 1(dy/dx = 1)이니까 그대로...
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # 예의 그 시그모이드 결과
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # 시그모이드 미분이 y(1-y) 이므로...
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        # W와 b는 초기화에 받는다..이건 고정이란 얘기...
        self.W = W
        self.b = b
        # 입력 x는 dW 미분 계산에 필요하니까 선언해 두는데...나머지는?
        self.x = None
        self.orgin_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 첫 번째 차원 크기로 놓고 나머지는 펴버린다...뭐 하는지는 알겠는데, 왜 이렇게 하는지는...
        # 아마도 텐서 계산할 때는 W나 b를 텐서에 맞게 늘려놔서 x도 펴서 곱해야 하나?
        self.orgin_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        # 원소 수에 맞게 원래 shape로 변경한다...
        dx = dx.reshape(*self.orgin_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        # 출력이 소프트맥스가 아니라 손실함수 값인가?
        return self.loss

    # 우선 이게 최종이니 dout 값이 1로 고정이고(dL/dL)
    def backward(self, dout=1):
        # 배치로 돌리면 x나 t나 배치 크기는 같으니, 받은 t에서 첫 번째가 배치 사이즈
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
