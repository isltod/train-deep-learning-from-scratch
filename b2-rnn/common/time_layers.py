from common.np import *
from common.layers import *
from common.functions import sigmoid


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        # 211쪽 식 5.10 RNN 계산식
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        # 근데 tanh가 활성화 층이 아니고 그냥 RNN에 포함되어 있네...
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    # RNN에서는 dn_next가 이전의 dout
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # tanh의 미분 = 1 - h_next^2
        dt = dh_next * (1 - h_next**2)
        # b에서 repeat 있으므로 합치기
        db = np.sum(dt, axis=0)
        # dWh는 반대쪽인 h 방향(전치) 곱
        dWh = np.dot(h_prev.T, dt)
        # dh는 반대쪽인 Wh 방향 전치 곱
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        # 뭔가 원칙이 params에 속하는 가중치/편향은 그냥 계산해놓고, 입력인 x/h는 반환하는 모양...
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        # Wx는 단어 표현 차원 D x 은닉 상태 차원 H, Wh는 HxH
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        # 여기 h는 이전 TimeRNN 덩어리에서 넘어올 상태 변수...
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        # N은 배치, T는 문장 내 단어 순서?, D는 단어 차원...
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        # Xs에서 단어차원 D만 은닉 차원 H로 바꾼 모양
        hs = np.empty((N, T, H), dtype="f")

        # 넘어온 상태라는 것이 없다면 h는 0
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        for t in range(T):
            # TimeRNN내의 모든 RNN들은 매개변수들이 같다...
            layer = RNN(*self.params)
            # 순서대로 t번 단어는 t번째 RNN에 넣는다...h는 0~T까지 계속 재귀 업데이트
            # 넘어온 상태 h가 없어도 여기서 업데이트하면 넘어갈 상태 h는 있다는 얘긴데...
            self.h = layer.forward(xs[:, t, :], self.h)
            # 그리고 업데이트 된 h는 다시 t번째 Hs에 넣어둔다...
            hs[:, t, :] = self.h
            # layer도 순서대로 붙여서 저장...
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        # TimeRNN에서는 이전의 dout가 dhs
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0]
        # 역전파 반복 순서는 순전파의 역순
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 일단 209쪽 그림 5-15 참고해서...
            # 역전파는 이전 RNN(TimeRNN 마찬가지)에서 넘어오는 dh, t 순번대로 위에서 내려오는 dhs 입력해서
            # 이전 RNN으로 넘기는 dh, 아래로 내리는 dx가 나오는데...
            # h는 순전파에서 위와 오른쪽을 분기했으므로 더하고, RNN 내부적인 역전파를 거쳐 dh, dx 나온다...
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            # 위에 grads는 총 3개로 초기화됐고, 위 RNN 클래스에서 grads는 dWx, dWh, db 세가지를 담는데...
            for i, grad in enumerate(layer.grads):
                # grads[0]에는 0~T-1번째 dWx가 다 합쳐져 들어간다...
                grads[i] += grad

        # T-1~0번째 dWx를 다 합친걸 self.grads[0]에, dWh 다 합친건 self.grads[1]에 덮어쓰기라...
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        # T-1부터 0까지 RNN 역전파 거친 dh가 최종 dh...
        # 근데 BPTT에서는 이걸 쓰지 않는다고...일단 backward의 인수에도 이전 TimeRNN에서 내려오는 dh는 없다...
        self.dh = dh

        return dxs


class TimeEmbedding:
    def __init__(self, W):
        # W는 단어 사전 크기 V x 단어 표현 차원 D
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        # Xs는 모양이 NTD 아닌가?
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype="f")
        self.layers = []

        # 0~T-1개 단어어 대해서
        for t in range(T):
            # 단어별로 Embedding 층을 만들어서, t번째 단어를 한 번에 넣어서 출력을 만든다...
            # 근데 그게 N1D모양인가? 그리고 그걸 합쳐서 NTD 모양 행렬로 만든다?
            layer = Embedding(self.W)
            # 인덱스들을 받아서 위에 넣은 W의 인덱스에 해당되는 행들을 꺼내는 과정인데...
            # xs[:, t]가 왜 그런 인덱스가 되는거지?
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        # RNN 아니면 순서 상관없어서 reverse 안하나? 0~T-1까지 돌면서
        for t in range(T):
            layer = self.layers[t]
            # 순전파에서 넘겨준 인덱스에 해당되는 dout의 행? 열? 들을 누적합하는 과정인데...
            # 위에서 내려온 역전파 미분 행렬 중, 순전파에서 처리해서 넣었던 t 부분만 역전파 따로...
            layer.backward(dout[:, t, :])
            # 그렇게 얻은 기울기는 계속 더한다...
            grad += layer.grads[0]
        # 여긴 self.grads의 원소가 dW 하나 뿐이라서 [0]은 별 의미는 없는데...
        self.grads[0][...] = grad
        return None


# 근데 TimeAffine은 TimeRNN이나 TimeEmbedding 처럼 레이어들을 만들어서 반복하지 않고,
# 그냥 한 방에 묶어서 계산하네...올라올 때도, 위로 전달할 때도, 또 반대방향도 이렇게 처리가 되나?
class TimeAffine:
    def __init__(self, W, b):
        # TimeAffine에서 같이 묶여있는 Affine들은 가중치가 다 같나?
        # W는 은닉 상태 차원 H x 단어 사전 크기 V
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        # 배치 수, 단어 순서는 하나의 축으로 묶어서 행렬 변경...이러면 W 형태를 여기 맞춰서 주나?
        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        # 내보낼 때는 다시 NTD 형태로 변경
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        # 여기서도 위에서 내려온 미분값과 x를 (배치축 x 단어순서, 단어차원) 형태로 변경해서 계산...
        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        # 이렇게 되면 (NT, D) x (D, H) = NTH?
        dW = np.dot(rx.T, dout)
        # 이건 (NT, H) x (H, D) = NTD?
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        # 역시나 매개변수는 여기에 저장하고 내려갈 dx만 반환?
        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        # 뭘 무시한다는 건지..
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 원-핫 벡터인 경우, 마지막 차원이 단어 표현...그걸 없애고...
            ts = ts.argmax(axis=2)

        # ignore_label에 표시된 값들은 무시하겠다?
        mask = ts != self.ignore_label

        # 배치용과 시계열용을 합쳐서 처리
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        # softmax는 import 안했지 않나?
        # 배치에 단어 순서까지 다 묶어서 소프트맥스? 그럼 분모가 다 합쳐진건데...
        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        # 어쨌든 이렇게 통짜로 나누면 단어당 평균 손실이 되긴 하겠지...
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys
        # dx = y - t
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        # 224쪽 그림 5-29 참고, 순전파에서 손실을 구할 때 다 더하고 1/T를 곱해줬으니...
        # 역전파에서는 일단 반대편 1/T를 곱해주고, 합 노드는 같은 값을 그대로(repeat) 넘긴다..
        dx /= mask.sum()
        dx = dx.reshape((N, T, V))

        return dx
