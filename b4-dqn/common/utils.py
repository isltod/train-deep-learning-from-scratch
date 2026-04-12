import matplotlib.pyplot as plt
import numpy as np


def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    # 이건 왜 필요한지...
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        # 이런 경우가 있나? max가 없어?
        return np.random.choice(len(xs))
    return np.random.choice(idxes)


def greedy_probs(Q, state, epsilon=0, action_size=4):
    # 현재 상태에서 모든 액션의 Q값을 가져옴
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = argmax(qs)

    # 기본적으로 모든 행동에 epsilon / action_size 만큼의 확률 배분
    base_prob = epsilon / action_size
    # 그럼 ε 만큼의 확률이 남는데...
    action_probs = {a: base_prob for a in range(action_size)}
    # 그걸 최적 행동에 추가...
    action_probs[max_action] += 1 - epsilon
    return action_probs
