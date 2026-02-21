import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # 이걸 쓰니까 딕셔너리인데도 순서가 유지된다...
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        # 배치? 텐서? 처리를 위해 axis 0은 배치?이고 답은 axis 1에 있는 모양...
        y = np.argmax(y, axis=1)
        # 근데 왜 t만 차원 검사를 해야 하나?
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        # 그래서 y, t가 1차원 벡터로 나오는 모양...평균 구하기...
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 이건 너무 느려서 실제로 사용하기 불가능했었는데...
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        # 손실값 - 이게 forward 기능에 해당하는건가...
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 그래프가 아니라 각 가중치와 편향 그래디언트를 딕셔너리로 내보내기 위해서 dW와 db가 필요했나?
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads
