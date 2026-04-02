from common.np import *  # import numpy as
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        # 왜 W(넘파이 행렬), grad를 다시 리스트 안에 넣는 걸까?
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        # 그걸 다시 튜플로? 이렇게 하면 리스트 안의 W를 튜플 안의 W로 받을 수 있는 모양인데...
        (W,) = self.params
        out = np.dot(x, W)
        # 입력 x는 역전파에서 dW 구하기 위해 저장...
        self.x = x
        return out

    def backward(self, dout):
        (W,) = self.params
        # 곱 역전파에서 x 방향은 W 전치를 곱하고, W 방향은 x 전치 곱하기...
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # grad 리스트의 첫 번째 요소(위에서 W 모양의 0 행렬)의 각 위치[...]에 dW 원소 넣기?
        self.grads[0][...] = dW
        # 근데 역전파에서는 dx만 반환하던가? dW는 위에 grad 리스트 안의 원소로 사용하나?
        return dx


class Affine:
    def __init__(self, W, b):
        # 일단 1권에서는 딕셔너리를 사용했는데, 그걸 리스트로 처리한다고 이해하자...
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        # 튜플로 안 받고 이렇게 해도 되는데, 위에서는 리스트 원소가 한 개라서 그런건가?
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        # 예의 그 역전파 미분값 구하기 - 곱은 반대로, 합은 그대로
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        # 경사도는 리스트로 저장해두고, 그 중 dx만 반환...
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        # 왜 functions에 함수 만들어놓고 안쓰는 거지?
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # 시그모이드 역전파 미분은 y(1-y)
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        # 그냥 CEE가 아니라 이진 교차 엔트로피(Binary Cross Entropy)여야 역전파 미분이 y - t로 간단하게 나온다...
        # BCE = -[tlogy + (1-t)log(1-y)]
        # np.c_는 두 열벡터를 이어붙여 행렬로 만들기...
        # 이 부분은 정답부분 예측과 오답부분 예측을 나눠서 각각 호출하고,
        # 정답부분에서는 t가 모두 [1, 1, ...], 오답부분에서는 모두 [0, 0, ...]으로,
        # 미리 t 또는 1-t 형태로 넘어오니까 아래 인수 부분에 1-t는 필요없음
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 위에서 BCE로 손실 값을 구하면, 역전파 미분은 y - t가 된다...나머진 위에서 흘러온 값과 평균 만들기...
        dx = (self.y - self.t) * dout / batch_size
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        # 순전파는 쉬운데...그냥 소프트맥스 함수 적용...
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        # Softmax 미분 수식과 코드 참고...
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        # 이건 소프트맥스 손실 함수니까 정답지를 받는다...
        self.t = t
        self.y = softmax(x)

        # 정답지가 소프트맥스와 같다면 원-핫 벡터, 다른 계산을 위해 argmax 인덱스 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        # 정답지 첫번째 차원이 배치...
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        # Softmax + Loss(CEE) 역전파 미분은 y - t = softmax - t,
        # 따라서 위에서 predict까지 한 벡터에서, argmax 정답 인덱스 부분에 1 빼주기...
        dx[np.arange(batch_size), self.t] -= 1
        # 위에서 흘러온 값 곱하는데, 보통은 1일 것이고...
        dx *= dout
        # 배치 크기로 나눠 평균값 구하기...
        dx /= batch_size

        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        # 훈련에서는 드롭아웃 비율에 따라 마스킹하는건 이해가 되는데...
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 훈련 아닐 때, 이건 뭐지? 입력 x에서 그냥 1 - 드롭아웃 비율 숫자를 빼는건 뭐냐?
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


# 이건 어휘가 너무 많을 때, 원핫 벡터가 문제가 되니까, 원핫 없이 가중치 처리하기 위한 노드...
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        # 이게 헛갈리게 튜플로 받는데, 위에 init보면 params에는 W 딱 하나...그러니까 params 리스트 벗기고 W 배열 받기
        (W,) = self.params
        # 추출할 행의 인덱스들이 배열(배치 처리)로 들어있다고 가정...역전파에서 쓰도록 저장
        self.idx = idx
        # 그럼 np.matmul 없이 간단히 슬라이싱으로 원핫 x 가중치 효과를 낼 수 있다...
        out = W[idx]
        return out

    def backward(self, dout):
        # 여기도 헛갈리지만, init보면 zeros_like가 배열 만들고 그걸 []로 다시 감쌌으니,
        # 바깥 괄호 벗기고 안쪽 배열 받는걸 튜플로 받는 코드로 만들었다..
        (dW,) = self.grads
        # 일단 0으로 채우고 위에서 내려온 dout를 순전파에서 추출된 행에 넣어준게 역전파 미분이다...
        # 순전파가 x*W인데 x가 원핫 벡터니까 그 자리에 1 곱하기가 미분이란 말인가?
        # 층이 둘 밖에 없는 구조라서 x 미분은 쓸데 없고?
        dW[...] = 0
        # 뭔가 예전(8이하)에는 add.at을 호출해 scatter_add를 연결해서 처리했는데, 이게 없어졌다고...
        # 지금은 cupyx에 scatter_add 함수를 사용한다고...
        if GPU:
            import cupyx

            # 하는 일은 아래 else 부분 참고...
            cupyx.scatter_add(dW, self.idx, dout)
        else:
            # idx 배열마다 돌면서 dW의 idx 자리에 dout를 더한다...아래와 같은 코드...
            np.add.at(dW, self.idx, dout)
            # for i, word_id in enumerate(self.idx):
            #     dW[word_id] += dout[i]
        return None
