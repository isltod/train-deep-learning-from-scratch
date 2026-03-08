import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        # vocab_size가 단어 원핫 벡터 차원, hidden_size는 은닉층 차원
        V, H = vocab_size, hidden_size

        # 가중치 초기화 - 32비트 소수로 난수
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        # 입력층 2 - window 1
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        # 출력층(이게 디코더?)과 손실함수(소프트맥스와 크로스 엔트로피) 층은 분리...
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            # 더하기를 하면 append가 되나?
            self.params += layer.params
            self.grads += layer.grads

        # 입력 가중치와 출력 가중치가 결국 단어의 분산표현 벡터...그 중 입력 가중치 이용
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # 입력은 cnn과 다르게 contexts에서 뽑아 쓰는데, 여기는 두 단어
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        # 평균내고
        h = (h0 + h1) * 0.5
        # 일단 행렬곱으로 출력 만들고, 소프트맥스와 손실값은 추가로 구하기...
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        # 역전파는 순서대로 backward를 호출하는데...
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        # 중간에 평균내려고 0.5 곱했던 부분 추가, 덧셈은 그대로니까 da를 양쪽으로...
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
