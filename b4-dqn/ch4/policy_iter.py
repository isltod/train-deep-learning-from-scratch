from collections import defaultdict
from policy_eval import policy_eval
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld


def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}
    # 그리드월드의 모든 격자에 대해서
    for state in env.states():
        # 가능한 액션에 대한 가치값을 딕셔너리로 구하고...
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 여기서 가치함수 V는 policy_eval에서 반복 갱신 방법으로 얻어온다...
            value = r + gamma * V[next_state]
            action_values[action] = value
        # 그 중 최대 가치를 확률 1로...무조건 선택이라 greedy...
        max_action = argmax(action_values)
        action_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        action_probs[max_action] = 1.0
        # 그걸 격자별로 저장해서 반환...
        pi[state] = action_probs
    return pi


def policy_iter(env, gamma=0.9, threshold=0.0001, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)
    while True:
        # 정책 평가 - 현재 π를 기반으로, 반복 갱신으로 가치함수 V를 얻고
        V = policy_eval(pi, V, env, gamma, threshold)
        # 탐욕화 - 새로 얻은 V에서 가장 좋은 것만 취하는 정책으로 바꾸고...
        new_pi = greedy_policy(V, env, gamma)
        if is_render:
            env.render_v(V, pi)
        # 기존 정책이나 새 정책이 같다면 이미 최적 정책
        # 근데 해석적으로는 맞아도 수치적으로 맞나? 몇 번 더 해보면 달라질 수 없나?
        if pi == new_pi:
            break
        pi = new_pi
    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
