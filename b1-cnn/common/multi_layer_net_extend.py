import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNetExtend:
    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
        use_dropout=False,
        dropout_ratio=0.5,
        use_batchnorm=False,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 계층 만들기
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )
            if self.use_batchnorm:
                self.params["gamma" + str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params["beta" + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers["BatchNorm" + str(idx)] = BatchNormalization(
                    self.params["gamma" + str(idx)], self.params["beta" + str(idx)]
                )

            self.layers["Activation_function" + str(idx)] = activation_layer[
                activation
            ]()

            if self.use_dropout:
                self.layers["Dropout" + str(idx)] = Dropout(dropout_ratio)

        # 배치 정규화나 드롭아웃 경우에는 마지막에는 사용하지 않는 모양...
        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(
            self.params["W" + str(idx)], self.params["b" + str(idx)]
        )

        self.last_layer = SoftmaxWithLoss()

    # 가중치 초기값을 정규분포 난수로 만들고, 활성화 함수 종류에 따라 초기값 조절해서 반환
    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params["W" + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx]
            )
            self.params["b" + str(idx)] = np.zeros(all_size_list[idx])

    # predict라는건 결국 마지막 softmax, loss 빼고 나머지 노드들 순서대로 곱해서 반환...
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    # 결국 loss라는건 predict에서 뺀 마지막 노드(소프트맥스에 크로스 엔트로피)에 예측치와 정답지 적용해 반환
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        # 여전히 weight_decay 잘 모르겠고...
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = numerical_gradient(
                loss_W, self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = numerical_gradient(
                loss_W, self.params["b" + str(idx)]
            )

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads["gamma" + str(idx)] = numerical_gradient(
                    loss_W, self.params["gamma" + str(idx)]
                )
                grads["beta" + str(idx)] = numerical_gradient(
                    loss_W, self.params["beta" + str(idx)]
                )

        return grads

    # gradient가 실제로 쓰이는 역전파인데...Loss계산해놓고, 층마다 돌면서 backward 계산시키는 것...
    # 이 자체로는 반환값이 아니고, 그 과정에서 각 노드별 dW, dx, db 등을 계산시켜 놓는게 중요하다
    # 결국 그 미분값, 즉 노드별 그라디언트는 grads 딕셔너리에 넣어서 반환 - 이걸로 노드별 가중치 갱신...
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = (
                self.layers["Affine" + str(idx)].dW
                + self.weight_decay_lambda * self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads["gamma" + str(idx)] = self.layers["BatchNorm" + str(idx)].dgamma
                grads["beta" + str(idx)] = self.layers["BatchNorm" + str(idx)].dbeta

        return grads
