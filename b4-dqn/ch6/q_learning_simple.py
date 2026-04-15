from collections import defaultdict, deque
import numpy as np
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        # ε 확률로 무작위 탐색
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # 1-ε 확률로는 탐욕(최대화 결정론적) 행동
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        # update는 maxQ 이용해서 Q 함수만 갱신...
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            if done:
                break
            state = next_state

    env.render_q(agent.Q)
