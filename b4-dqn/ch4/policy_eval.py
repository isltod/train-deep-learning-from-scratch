import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld
from collections import defaultdict


env = GridWorld()
# 이게 아래처럼 초기화하지 않아도 되는 코드...
V = defaultdict(lambda: 0)
# V = {}
# for state in env.states():
#     V[state] = 0

state = (1, 2)
print(V[state])

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
state = (0, 1)
# 여기서 없던 (0,1) 키를 넣어서 조회하는 순간, 위에 람다식으로 만든 딕셔너리가 pi[(0,1)]에 할당된다...
print(pi[state])


def eval_onestep(pi, V, env, gamma=0.9):
    # 그리드월드의 모든 격자에 대해서
    for state in env.states():
        # 목표 지점이면 해당 상태 가치 함수는 0, 나머진 계산 없이 통과
        if state == env.goal_state:
            V[state] = 0
            continue

        # 해당 격자의 각 행동 확률(딕셔너리)에 대해서
        action_probs = pi[state]
        new_v = 0
        for action, action_prob in action_probs.items():
            # 현재 상태와 가능한 각 행동에 대한 다음 상태, 그에 따른 보상
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 다음 상태에 대한 확률 곱해서 가치함수 누적 갱신...이게 한 번만 한다고 onestep...
            new_v += action_prob * (r + gamma * V[next_state])
        V[state] = new_v
    return V


def policy_eval(pi, V, env, gamma=0.9, threshold=0.0001):
    while True:
        old_V = V.copy()
        # 한 번 갱신해보고
        V = eval_onestep(pi, V, env, gamma)

        # 모든 상태에 대해서 V의 최대 갱신값을 찾아서 그걸 δ로...
        delta = 0
        for state in V.keys():
            t = abs(old_V[state] - V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            # 이런 δ가 0.0001보다 작다면 그만 갱신...
            break
    return V


# pi, V, env는 위에서 다 선언해놨고...
gamma = 0.9
V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)
