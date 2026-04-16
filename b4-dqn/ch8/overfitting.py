"""
먼저 DQN TD 목표가 과대적합 되는 경향이 있다고 할 때의 과대적합은 일반적인 딥러닝 모델의 과대적합과는 다르다.
모델이 훈련 데이터만 너무 잘 맞추는 문제가 아니라, 목표 값을 더 크게 예측하는 것이다.
q(s,a0) = q(s,a1) = q(s,a2) = q(s,a) = 0이면
E[max{q(s,a)}] = 0이다.
하지만 Q(s,a) ~ q(s,a) + ε(random noise가 섞여있다면)라면
E[max{Q(s,a)}] = E[max{q(s,a)}] + max(ε) > 0이므로
E[max{Q(s,a)}] > E[max{q(s,a)}]이 되는 문제이다.
"""

import numpy as np
import matplotlib.pyplot as plt


def draw_hist(data):
    plt.hist(data, bins=16)
    plt.axvline(x=0, color="red")
    data = np.array(data)
    plt.axvline(data.mean(), color="cyan")
    plt.show()


# 표본 1000개로 실습...
samples = 1000
action_size = 4
Qs = []

for _ in range(samples):
    # 원래 q에 정규분포를 따르는 무작위 노이즈를 추가해서 Q 생성
    Q = np.random.randn(action_size)
    Qs.append(Q.max())

# 히스토그램 비교
draw_hist(Qs)

"""
    이런 과대적합을 해소하기위해 Double DQN을 사용한다.
"""

Qs = []

for _ in range(samples):
    # 두 개의 표본 집단이 있고, 서로 독립이라면
    Q = np.random.randn(action_size)
    Q_prime = np.random.randn(action_size)

    # 이렇게 한 쪽에서 max 인덱스를 봐서 그 위치에 해당되는 데이터는 다른 집단에서 뽑는다면...
    idx = np.argmax(Q)
    Qs.append(Q_prime[idx])

# max는 Q에만 있는 거라서, Q_prime에서 뽑은 것들의 오차 기댓값은 0이 된다...
draw_hist(Qs)
