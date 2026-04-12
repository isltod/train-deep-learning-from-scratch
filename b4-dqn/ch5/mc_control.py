from collections import defaultdict
import numpy as np
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld
from common.utils import greedy_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1  # ε-greedy를 위한 확률
        self.alpha = 0.1  # 증분 갱신을 위한 학습률
        # self.cnts = defaultdict(lambda: 0)
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            # V 갱신과 다른 건 key가 state 하나가 아니라 (s, a)라는 것...
            key = (state, action)
            # Q[s, a]를 증분식으로 갱신 (고정 학습률 alpha 사용)
            # self.cnts[key] += 1
            # self.Q[key] += (G - self.Q[key]) / self.cnts[key]
            self.Q[key] += (G - self.Q[key]) * self.alpha

            # ε-greedy 정책 개선
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == "__main__":
    env = GridWorld()
    agent = McAgent()

    # 만 번을 시도하는데...각각이 무한루프를 돌면서 종점에 도달하면 끝...
    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            if done:
                agent.update()
                break
            state = next_state

    env.render_q(agent.Q)
