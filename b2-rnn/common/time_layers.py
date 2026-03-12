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
                # TimeRNN 안에는 T개의 RNN이 같은 가중치를 가지고, 이건 Wx 등이 분기해서 들어간 꼴...
                # 그래서 클래스별로 다 합쳐저야 되고, grads[0]에는 0~T-1번째 dWx가 다 합쳐져 들어간다...
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


class LSTM:
    def __init__(self, Wx, Wh, b):
        # Wx는 단어 표현 차원 D x 은닉 상태 차원 H*4 - Wf, Wg, Wi, Wo가 연달아 묶인 형태
        # Wh는 H x H*4, b는 H*4
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # 257쪽 식 6.6 참고, A에는 f = XtWf + Bf, g = XtWg + Bg, ... 네가지의 계산 결과가 이어져 있고...
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # 각각이 은닉 상태 차원 H 만큼씩 결과를 내니까, 행 배치는 전부, 열은 H씩 끊어서 f, g, i, o
        f = A[:, :H]
        g = A[:, H : 2 * H]
        i = A[:, 2 * H : 3 * H]
        o = A[:, 3 * H :]

        # g만 메모리 값(-1~1), f/i/o는 다 게이트(0~1), 그에 맞게 활성화 함수 적용
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        # 255쪽 그림 6-18 참고, c_next는 f와 아다마르 곱, g와 i도 아다마르 곱, 둘을 합친다...
        c_next = f * c_prev + g * i
        # 그 결과에 tanh 적용하고 출력 게이트와 아다마르 곱하면 다음 상태값 h
        h_next = o * np.tanh(c_next)

        # 역전파 계산에 필요한 값들을 저장? h와 c next는 반환...
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    # 259쪽 그림 6-21 참고, 우선 위에서 넘어오는 dout는 dh_next와 dc_next 두 가지로 오고...
    def backward(self, dh_next, dc_next):
        # f, g, i, o에 대한 가중치들이 연결된 행렬 형태...
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)
        # dh_next * o -> h_next 방향에서 dh_next가 넘어오면 곱 노드는 반대 쪽 o를 곱(아다마르)해서 역전파
        # 1 - tanh_c_next**2 -> 위에서 받은 미분에 tanh의 미분 1 - y^2 (y=tanh[c_next]) 아다마르 곱해서 다시 전달
        # dc_next -> Ct가 Ct와 Ht로 분기했으므로 둘을 더하기
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next**2)

        # g x i와 덧셈은 그냥 흐르니 무시하고, f와 곱은 반대 측 값을 아다마르 곱해서 역전파 전달
        dc_prev = ds * f

        # o 방향 미분은 tanh와 아다마르 곱이었으므로, 위에서 넘어온 dh_next에 반대측 tanh_c_next 곱해서 전달
        do = dh_next * tanh_c_next
        # 다음은 sigmoid 함수 거쳤으므로 do * o(1-o) 아다마르 곱
        do *= o * (1 - o)

        # i와 g 방향은 Ct 경로로 넘어온 ds에, Ct와 합은 무시(곱 1)하고,
        # 각각 서로 아다마르 곱했었으므로 반대측 (i->g, g->i)를 곱해서 전달
        di = ds * g
        dg = ds * i
        # 그 다음은 i는 sigmoid, g는 tanh 거쳤으므로 각각 y(1 - y), 1 - y^2 아다마르 곱해서 전달
        di *= i * (1 - i)
        dg *= 1 - g**2

        # f 방향 미분도 넘어온 ds에 반대측 c_prev 아다마르 곱, 그 다음엔 sigmoid 부분 반영
        df = ds * c_prev
        df *= f * (1 - f)

        # 네 개로 slice 한 부분은 다시 모으는 것으로 역전파라고...
        # 세로가 배치, 가로가 출력이었으니가 가로 모으기
        dA = np.hstack((df, dg, di, do))

        # 맨 처음 4개 모은 Affine 변환 부분, 예의 반대측 전치 행렬 곱하기와 더하기...
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        # self.grads[0]에는 dWx, 1에는 dWh, 2에는 db 계속 덮어쓰기...
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        # 다음으로 넘기기 위해서 반환할 dx, dh_prev도 예의 그 Affine 변환 역전파...
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        # Wx는 단어 표현 차원 D x 은닉 상태 차원 H*4 - Wf, Wg, Wi, Wo가 연달아 묶인 형태
        # Wh는 H x H*4, b는 H*4
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        # 이걸 왜 인스턴스 변수로 저장하지? 이전 반복이 아니라 이전 단계의 TimeLSTM 값을 받아와야 하는거 아닌가?
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        # forward 호출할 때마다 레이어도 hs도 다 새로 만든다...이게 맞나?
        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = LSTM(*self.params)
            # 0~T-1번 레이어까지 순서대로 돌아가면서 앞의 h, c 받아서 다음 h, c 반환...
            # 이건 인스턴스 변수로 저장하지 않아도 되는거 아닌가?
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            # 또한 그 중 h는 hs 만들기
            hs[:, t, :] = self.h

            # 이건 역전파에서 쓸려고 저장? forward가 다시 호출되면 이건 다 초기화되는데...
            self.layers.append(layer)

        return hs

    # 이 역전파에서는 dout은 dhs
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        # Wx는 단어 표현 차원 D x 은닉 상태 차원 H*4
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype="f")
        dh, dc = 0, 0

        # self.grads[0]에는 dWx, 1에는 dWh, 2에는 db
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # h가 분기해서 hs의 t 자리에 들어가고, 다음 RNN으로 넘겨지므로 미분은 둘을 합하고...
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            # 거기서 나온 dx는 dxs의 t 자리에...순전파에서 xs의 t 자리가 입력됐으므로...
            dxs[:, t, :] = dx
            # dWx, dWh, db는 LSTM 레이어의 인스턴스 변수로 저장되어 있던 것들을 레이어 역순으로 다 더한다...
            for i, grad in enumerate(layer.grads):
                # TimeLSTM 안에는 T개의 LSTM이 같은 가중치를 가지고, 이건 Wx 등이 분기해서 들어간 꼴...
                # 그래서 클래스별로 다 합쳐저야 되고, grads[0]에는 0~T-1번째 dWx가 다 합쳐져 들어간다...
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None
