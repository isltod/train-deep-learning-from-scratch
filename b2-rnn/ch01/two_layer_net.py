import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# import numpy as np
from common.np import *
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 입력, 은닉, 출력 크기에 맞춰 2층 가중치와 편향 랜덤하게...
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # 2층과 출력층 만들고 - grads는 각 노드에서 0으로 초기화...
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]
        self.loss_layer = SoftmaxWithLoss()

        # 근데 각 층의 가중치, 편향, 기울기를 다 여기서 모아서 저장? 왜지?
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # 예측값과 손실까지를 구분해서, 예측은 predict, 손실까지는 forward로 나눈다고...
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    # 역전파는 backward 하나로, 각 노드별로 다 backward...
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
