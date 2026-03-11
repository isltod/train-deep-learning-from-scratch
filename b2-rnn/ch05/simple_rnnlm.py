import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.time_layers import *


class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        # 단어 사전 크기, 단어 표현 차원, 은닉 상태 벡터 차원..
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화 - 일단 같은 그룹 안에 있는 것들은 가중치가 같다...
        # embedding은 가중치만, 나머지는 편향이 있기는 한데, 일단 다 0이다..
        embed_W = (rn(V, D) / 100).astype("f")
        # 이전 계층 노드 수(앞쪽 차원)를 제곱근으로 나누는 방식은 Xavier 초기값...
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype("f")
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype("f")
        rnn_b = np.zeros(H).astype("f")
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            # Truncated BPTT는 작은 단위 여럿으로 나눠서 붙이겠다는 의미니까,
            # 순전파에서 h를 연결해줄 stateful=True가 필요하다...
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 모든 가중치와 기울기를 리스트에 모음 - 이건 왜 이렇게 하는지 아직도 잘 모르겠는데...
        # 아무튼 이것도 역전파에서 같은 그룹 안에 같은 가중치는 제외하고 그 기울기는 대표에게 몰아주겠지..
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # 순전파 역전파는 그냥 단순하게 각 레이어들 순서대로 forward, backward 호출...
    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()
