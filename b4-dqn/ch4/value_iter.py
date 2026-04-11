from collections import defaultdict
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld
from policy_iter import greedy_policy


def value_iter_onestep(V, env, gamma):
    # 모든 격자들에 대해서 돌아가면서...
    for state in env.states():
        # 그 격자가 목표점이면 목표점에 대한 가치함수 0, 나머진 계산안하고 통과...
        if state == env.goal_state:
            V[state] = 0
            continue

        # 그 외의 지점들에 대해서는 가능한 모든 이동 방향에 대해서
        action_values = []
        for action in env.actions():
            # 특정 방향으로 이동에 대한 상태, 보상, 가치 기댓값 순서로 구해서
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        # 이번 상태에서의 최대 가치 기댓값을 이번 상태의 기댓값으로 고정...
        V[state] = max(action_values)
    return V


# 요건 앞의 정책 반복법의 policy_iter와 같은데...
def value_iter(V, env, gamma, threshold=0.001, is_render=False):
    while True:
        if is_render:
            env.render_v(V)

        # 갱신전 갱신후 비교위해 기존값 저장하고 갱신
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        # 모든 상태에서 최대로 개선된 값을 찾고
        delta = 0
        for state in V.keys():
            t = abs(old_V[state] - V[state])
            if delta < t:
                delta = t
        # 최대 개선 값이 threshold를 못 넘으면 최적 상태...
        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    V = defaultdict(lambda: 0)
    # 이게 상태 가치 기댓값 갱신이고
    V = value_iter(V, env, gamma, is_render=True)
    # 이게 탐욕화 선택이고...
    pi = greedy_policy(V, env, gamma)
    # 정책 반복에서는 iter - greedy 루프를 계속 도는데, 여기서는 iter 혼자 돌고, greedy는 마지막에 확인용...
    env.render_v(V, pi)
