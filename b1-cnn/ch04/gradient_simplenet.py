import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # 표준 정규분포 난수 생성해서 2x3 2차원 배열로
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        # 1x2 X 2x3 -> 1x3 행렬곱
        return np.dot(x, self.W)

    def loss(self, x, t):
        # 입력 받아서 랜덤으로 만든 가중치 적용한 걸 predict라고...
        z = self.predict(x)
        # 소프트맥스 적용해서 출력층으로
        y = softmax(z)
        # 출력층 결과 Y를 정답지 t와 비고해서 손실 측정
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == "__main__":
    # 정답지
    t = np.array([0, 0, 1])

    # 입력
    x = np.array([0.6, 0.9])

    # 신경망
    net = simpleNet()

    f = lambda w: net.loss(x, t)
    dw = numerical_gradient(f, net.W)
    print(dw)
