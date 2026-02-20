import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        # 가중치는 표준 정규분포 난수를 히든 크기만큼 만들고, weight_init_std는 뭐냐?
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        # 예측이란 x*W + b, 항상 똑같고...
        a1 = np.dot(x, W1) + b1
        # 히든에 대한 활성화 함수는 sigmoid, relu
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        # 출력에 대한 활성화 함수는 identity(회귀), softmax(분류)
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        # 손실함수도 CEE 항상 똑같고...
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        # 뭔가 이 헷갈리는 더미 변수 W 넣는 스타일도 전형인 모양인데...
        # 여기서도 가중치를 매개변수로 넘기는 건 아무 의미가 없고,
        # 아래 numerical_gradient에 함수에 추가로 매개변수로 넘기면 거기서 수정해서 돌리겠지...
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        # 여기도 헛갈리게 코딩해놨는데,
        # 위에 제목의 numerical_gradient는 TwoLayerNet.numerical_gradient,
        # 이 아래 numerical_gradient는 common.gradient.numerical_gradient
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params["W1"].shape)
print(net.params["b1"].shape)
print(net.params["W2"].shape)
print(net.params["b2"].shape)
