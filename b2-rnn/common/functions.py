from common.np import *  # import numpy as


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # 배치로 결과가 여러개 들어오면..
    if x.ndim == 2:
        # 지수가 너무 커서 오버플로 되는 문제 방지...keepdims는 기존 배열 차원 유지...
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        # 가로축(데이터)으로 계산해서 세로 축(배치) 그대로 유지...
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    # 아래 연산들을 위해서 벡터도 행렬로 변환
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 이 둘이 같으면 t가 원핫인코딩 상태이고, 그럼 그 중 최대값 인덱스만 뽑아서 다음에 사용...
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # 배치 사이즈대로 0~끝까지 돌면서, argmax로 얻은 최대값 인덱스로 원소 받아오기
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
