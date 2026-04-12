from collections import defaultdict
import numpy as np
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld
from common.utils import greedy_probs


class McOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.2
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 타깃 정책
        self.b = defaultdict(lambda: random_actions)  # 거동 정책
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.b[state]  # 거동 정책에서 행동 추출
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        rho = 1
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            key = (state, action)

            # 가중치 rho를 적용하여 Q 갱신
            self.Q[key] += (rho * G - self.Q[key]) * self.alpha

            # 타깃 정책(pi)을 탐욕적(ε=0)으로 개선
            self.pi[state] = greedy_probs(
                self.Q, state, epsilon=0, action_size=self.action_size
            )

            # 거동 정책(b)은 ε-greedy로 유지
            self.b[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)

            # 가중치 갱신 (중요도 샘플링) - 344쪽 식 A.4를 345쪽 방식으로 뒤에서부터 구해서 곱하기
            rho *= self.pi[state][action] / self.b[state][action]
            if rho == 0:
                break


if __name__ == "__main__":
    env = GridWorld()
    agent = McOffPolicyAgent()

    # 여기서도 만 번 도는데...각 에피소드에서 결승점 도달 때까지 무한 반복...
    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            # 결승점 도달하면 한 번에 목표 정책, 행동 정책, 가중치 갱신...
            if done:
                agent.update()
                break
            # 아니면 계속 반복
            state = next_state

    env.render_q(agent.Q)
