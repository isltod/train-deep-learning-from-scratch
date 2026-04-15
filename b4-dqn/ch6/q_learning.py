from collections import defaultdict, deque
import numpy as np
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld
from common.utils import greedy_probs


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            # 다음 상태에서 가질 수 있는 모든 Q값 중 최대값을 선택 (Off-policy)
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        # Q-러닝 갱신 식
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 정책 개선 (b는 ε-greedy, pi는 결정론적)
        self.pi[state] = greedy_probs(self.Q, state, 0, self.action_size)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # Q 업데이트하고 정책 개선...
            agent.update(state, action, reward, next_state, done)

            if done:
                break
            state = next_state

    env.render_q(agent.Q)
