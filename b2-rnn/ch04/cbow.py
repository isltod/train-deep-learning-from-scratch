import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.np import *
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 가중치 초기화 - 어휘 사전 크기 x 은닉층 크기
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # 계층을 만드는데, 달랑 입력 윈도우 크기 x 2(양쪽), 그리고 마지막 손실함수 레이어
        self.in_layers = []
        for i in range(2 * window_size):
            # wordvec 모델은 일단 입력에서 은닉으로 넘어가는 가중치가 다 같다...
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 매개변수를 모아두는데, 이건 왜 이렇게 모으는지 아직도 잘 이해가 안간다...
        # 이렇게 [] 형태로 더해야 한 번에 리스트로 묶인다..
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 일단 입력 가중치와 출력 가중치는 서로 다른데...이 둘 다 단어벡터라고 하는데...
        # 어쨌든 단어 벡터는 입력 매개변수를 사용한다...
        self.word_vecs = W_in
        self.word_vecs1 = W_out

    def forward(self, contexts, target):
        # h를 스칼라 0, 나중에 배열 더하면 배열이 된다...
        h = 0
        # target 단어를 예측할 맥락 단어들을 윈도우 맨 왼쪽부터 embdding 해서 평균내기...
        for i, layer in enumerate(self.in_layers):
            # contexts 행은 target, 열은 맨 왼쪽 윈도우 0, 그 다음 1...
            # 각 레이어가 맨 왼쪽부터 context 단어 하나씩을 처리해서 은닉층으로 보내기(합)
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        # 출력 가중치 embedding, dot, sigmoid, BCE 해서 손실 구하기...
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        # 역전파는 역순으로 backward 하는데...
        dout = self.ns_loss.backward(dout)
        # 합 자체는 그냥 흘려도 되지만, 평균내려고 1/n 곱했던 건 반대방향 곱해서 보내야 하니까...
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
