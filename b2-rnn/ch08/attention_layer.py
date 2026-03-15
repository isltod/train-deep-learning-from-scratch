import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.np import *
from common.layers import Softmax


class WeightSum:
    def __init__(self):
        # 밖에서 받는 hs, a, dc 외에 자체적으로 가지는 가중치는 없다...
        self.params, self.grads = [], []
        self.cache = None

    # Encoder 각 LSTM에서 만든 h들과 이들에 대한 확률 a 받아서...
    def forward(self, hs, a):
        N, T, H = hs.shape

        # 단어표현 차원으로 늘려서 모양 맞추고
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        # 이건 원소별 곱
        t = hs * ar
        # 단어 차원으로 합쳐서 확률 단어(확률이 높은 단어가 더 많이 기여한 합성 단어 표현) 만들기...
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        # 단어 차원 합치기에 대해서 분기해주고
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        # ar 방향은 hs, hs 방향은 ar 곱하기, 원래가 원소별 곱이었으므로 전치 행렬 아니고 아다마르 곱
        dar = dt * hs
        dhs = dt * ar
        # 확률 a를 단어 차원으로 분기한 것은 다시 합
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        # LSTM에서 내놓은 h를 hs에 있는 문장 내 단어 수 만큼(합, 날짜 문제는 문자 수) 분기해서 맞춰주고
        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        # 원소별 곱하고
        t = hs * hr
        # 단어별 유사도가 목표이므로 위 WeightSum과 달리 단어(문자) 차원이 아니라 각 단어별 표현 차원으로 합...
        # 그러면 단어별 유사도가 내적 형식으로 나온다...a1*x1 + a2*x2...
        s = np.sum(t, axis=2)
        # 마지막으로 소프트맥스로 정규화...
        a = self.softmax.forward(s)

        self.cache = (hs, hr, a)
        return a

    def backward(self, da):
        hs, hr, a = self.cache
        N, T, H = hs.shape

        # 이건 이해가 안되지만, 일단 소프트맥스 역전파를 돌리고
        ds = self.softmax.backward(da)
        # 내적을 위해 합했던 것을 단어 표현 차원으로 분기 반복
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        # 반대 방향 아다마르 곱
        dhs = dt * hr
        dhr = dt * hs
        # 처음에 단어 차원으로 분기했던 걸 합해서 처리
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class Attention:
    def __init__(self):
        # 여기도 학습할 가중치와 경사도는 없고...
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        # 위에 AttentionWeight에서 나온 확률 벡터
        self.attention_weight = None

    def forward(self, hs, h):
        # 단순히 AttentionWeight와 WightSum을 이어 붙이는 과정
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        # 결과로 나온 단어별 유사도 a는 일단 저장...어디서 쓰지?
        self.attention_weight = a
        return out

    def backward(self, dout):
        # WeightSum의 hs방향과 a 방향 역전파 미분값 받고
        dhs0, da = self.weight_sum_layer.backward(dout)
        # 그 중 a 방향은 다시 AttentionWeight의 LSTM h 방향과 hs 방향으로 받고
        dhs1, dh = self.attention_weight_layer.backward(da)
        # 분기해서 들어갔던 hs는 합쳐주고
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:
    def __init__(self):
        # 여전히 학습할 자체 가중치와 경사도는 없다...
        self.params, self.grads = [], []
        self.layers = None
        # Attention 노드에서 저장한 단어별 유사도 벡터를 모아 행렬로 관리
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        # 문장 내 단어 순, 또는 줄 단위에 문자 순으로 돌면서...
        for t in range(T):
            # 각각 어텐션 만들고
            layer = Attention()
            # 인코더에서 넘어온 hs(_enc)와 LSTM에서 만든 h(s_dec[t]) 넣고,
            # 단어 또는 문자 순서에 맥락 벡터를 넣어 행렬로 만든다...
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            # 레이어와 단어별 유사도 벡터를 모은 행렬 저장
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            # 단어 또는 순서대로 역전파...어차피 t 인덱스로 넣을테니까 reverse 필요없고...
            layer = self.layers[t]
            # 인코더에서 받은 hs 방향과 LSTM에서 받은 h 방향 역전파 미분값 받고
            dhs, dh = layer.backward(dout[:, t, :])
            # 인코더에서 받은 hs는 분기해서 넣었으니 다 합치고
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec
