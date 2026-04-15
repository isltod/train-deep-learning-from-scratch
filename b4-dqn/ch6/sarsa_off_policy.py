from collections import defaultdict, deque
import numpy as np
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld
from common.utils import greedy_probs


class SarsaOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 타깃 정책
        self.b = defaultdict(lambda: random_actions)  # 거동 정책
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        # pi가 아니라 b에서 행동 선택
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1
        else:
            next_q = self.Q[next_state, next_action]
            # 중요도 샘플링 가중치 rho = pi(a|s) / b(a|s)
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        # 오프-폴리시 SARSA 업데이트 식
        target = reward + self.gamma * next_q
        self.Q[state, action] += (rho * target - self.Q[state, action]) * self.alpha

        # 정책 개선 - pi는 탐욕화, b는 ε-탐욕화
        self.pi[state] = greedy_probs(
            self.Q, state, epsilon=0, action_size=self.action_size
        )
        self.b[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaOffPolicyAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)

            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state

    env.render_q(agent.Q)
