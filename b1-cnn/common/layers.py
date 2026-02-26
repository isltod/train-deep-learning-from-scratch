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
        # 입력 x는 dW 미분 계산에 필요하니까 선언해 두는데...
        self.x = None
        self.orgin_x_shape = None
        # 가중치와 편향 미분은 역전파 때 계산해서 저장한다...그래프 그릴 때 필요한가?
        self.dW = None
        self.db = None

    # 결국 foward가 하는 일은 정해진 xW + b 계산해 넘기기
    def forward(self, x):
        # 첫 번째 차원 크기로 놓고 나머지는 펴버린다...뭐 하는지는 알겠는데, 왜 이렇게 하는지는...
        # 아마도 텐서 계산할 때는 W나 b를 텐서에 맞게 늘려놔서 x도 펴서 곱해야 하나?
        self.orgin_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    # 결국 backward가 하는 일은 위에서 넘어온 미분 누적에,
    # 미리 계산해 놓은 현재 미분을 붙여서(곱은 반대 곱, 합은 그냥 보내기 등) 넘기기
    # xW + b에서 dL/dx = dL/dY . W_t 전치행렬로 곱해서 보낸다...
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        # 마찬가지로 dL/dW = x_t . dL/dY, x를 전치행렬로 앞에서 곱한다...
        self.dW = np.dot(self.x.T, dout)
        # db는 배치로 다 더했던 것을 하나로(세로로) 합친다...
        self.db = np.sum(dout, axis=0)
        # 원소 수에 맞게 원래 shape로 변경한다...
        dx = dx.reshape(*self.orgin_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    # 이 노드의 forward란 소프트맥스와 크로스 엔트로피 적용해서 반환
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        # 결국 출력은 소프트맥스가 아니라 손실함수, 이걸 이용하는 two_layer_net에서 loss를 필요로 한다...
        return self.loss

    # 소프트맥스 크로스 엔트로피 노드의 역전파 미분값은 y - t, 그걸 계산해서 넘겨주는게 backward
    # 우선 이게 최종이니 dout 값이 1로 고정이고(dL/dL), 근데 여기선 쓰지도 않는데?
    def backward(self, dout=1):
        # 배치로 돌리면 x나 t나 배치 크기는 같으니, 받은 t에서 첫 번째가 배치 사이즈
        batch_size = self.t.shape[0]
        # 정답지 t가 원핫인코딩이면...y도 t도 (batch_size, 10) 이런 모양
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            # 윈핫인코딩이 아니라면, 예측치를 dx로 넣고, 그 예측치에서 배치 0~n까지 정답 t 부분을 1을 뺀다..
            # 여기서 t는 원핫인코딩이 아니라 정답 위치 인덱스니까...
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            # 그리고 그걸 배치 사이즈로 나누면...나중에 다 더해야 평균되는거 아닌가?
            dx = dx / batch_size
        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # 입력 x의 shape 대로, 정규분포 값으로 행렬 만들고, dropout_ratio보다 큰 값만 살린다...
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    # 배치 정규화 forward는 정규화에 감마와 베타로 선형 변환한 값을 반환
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            # x가 2차원 아니면 4차원이란 얘기...shape 튜플의 각 원소를 N,C,H,W로 받는다...
            # 근데 쓰는건 N만? 그럼 그냥 x.shape[0]하면 되는거 아냐?
            N, C, H, W = x.shape
            # 그리고 여기서 입력 x의 shape를 막 바꿔도 뒤에 계산에 지장없나?
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    # 이게 배치 정규화 노드의 실제 foward,
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            # 여기 넘어올 때는 위에서 reshape해서 2차원으로 넘어온다...
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # 이 train_flg는 훈련과 검증이나 실제를 구분하나?
        # 아무튼 훈련이면 실행 평균/분산 구하고, 훈련 아니면 거기서 정규화를 구한다?
        if train_flg:
            # 예의 그 평균, 분산, 표준편차, 정규화 구하는 식
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            # 배치 크기, 오차, 표준편차, 정규화는 저장...
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        # 정규화에 감마, 베타 적용해서 최종 변환값이 forward...
        out = self.gamma * xn + self.beta
        return out

    # 배치 정규화 backward 수식은 모르겠고, 내려온 dL/dy에 분산, 표준편차, 오차 등을 버무려 dx 계산하고 반환
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    # 이게 실제 배치 정규화 backward인데 수식은 모르겠음...찾아보라고...
    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
