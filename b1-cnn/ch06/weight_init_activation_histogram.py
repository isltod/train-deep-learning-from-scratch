import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


if __name__ == "__main__":
    input_data = np.random.randn(1000, 100)
    node_num = 100
    hidden_layer_size = 5
    # 활성화 결과 저장
    activations = {}

    x = input_data

    for i in range(hidden_layer_size):
        if i != 0:
            # 이걸로 다음 층으로 넘기는 듯한 트릭...아래 가중치는 randn으로 계속 난수 적용...
            x = activations[i - 1]

        # 초깃값 바꿔가며 실험이라...
        # w = np.random.randn(node_num, node_num) * 1
        # w = np.random.randn(node_num, node_num) * 0.01
        # 이게 Xavier 초깃값...
        w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
        # 이건 He 초깃값
        # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

        a = np.dot(x, w)

        # 활성화 함수 바꿔가며 실험...
        # z = sigmoid(a)
        z = relu(a)
        # z = tanh(a)

        activations[i] = z

    for i, a in activations.items():
        # 세로 1줄, 가로 5개의 i + 1번째 조각 차트
        plt.subplot(1, len(activations), i + 1)
        plt.title(str(i + 1) + "-layer")
        # 맨 왼쪽 외에는 y 축 눈금과 숫자 없음
        if i != 0:
            plt.yticks([], [])
        # 이건 sigmoid용
        # plt.hist(a.flatten(), 30, range=(0, 1))
        # 이건 tanh용
        # plt.hist(a.flatten(), 30, range=(-1, 1))
        # 이건 relu용
        plt.hist(a.flatten(), 30, range=(0, 3))

    plt.show()
