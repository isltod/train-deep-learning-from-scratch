# import numpy as np
from common.np import *


def load_data(seed=1984):
    np.random.seed(seed)
    CLS_NUM = 3  # 클래스 수
    N = 100  # 클래스 당 샘플 수
    DIM = 2  #

    # 입력 X는 (3*100, 2), 정답지 t는 (3*100, 3) - 원핫 벡터
    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=int)

    for j in range(CLS_NUM):
        for i in range(N):
            # 회전율은 (0~99)/100 -> 0~1, 반지름도 그렇게...
            rate = i / N
            radius = 1.0 * rate
            # 각도 theta는 클래스 번호대로 4배해서, 회전율과 정규분포 난수 합에 더해지고...
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            # 클래스 수 고려한 인덱스 - 000 ~ 299
            ix = N * j + i
            # sin과 cos로 x, y 좌표?로 돌리고, radius 곱해서 원점에서 밖으로...flatten 해야 원래 2차원 유지...
            # 이거 원래 x가 rcosx, y가 rsinx일텐데...
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            # 클래스 번호대로 정답 원핫 인코딩
            t[ix, j] = 1

    return x, t
