import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 계층을 생성한다...
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )
            # 클래스 이름을 딕셔너리로 넣고, 키로 그걸 부른 후에, 뒤에 괄호 붙이면 초기화가 된다라...
            self.layers["Activation_function" + str(idx)] = activation_layer[
                activation
            ]()

        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(
            self.params["W" + str(idx)], self.params["b" + str(idx)]
        )
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            # 초깃값 스케일링,
            # 일단 기본값으로 넣어 놓는데, 값이 he나 xavier 아니면 0.01 같은 숫자...
            scale = weight_init_std
            # 그 외에 relu나 he 옵션을 주면 he 초깃값
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            # 아니고 sigmoid나 xavier 옵션에는 xavier 초깃값 설정
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params["W" + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx]
            )
            # 편향은 0으로 없고...
            self.params["b" + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(idx)]
            # 가중치 패널티 적용
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = numerical_gradient(
                loss_W, self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = numerical_gradient(
                loss_W, self.params["b" + str(idx)]
            )

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            # 여기서도 weight_decay가 들어가는 이유를 모르겠는데...
            grads["W" + str(idx)] = (
                self.layers["Affine" + str(idx)].dW
                + self.weight_decay_lambda * self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db

        return grads
