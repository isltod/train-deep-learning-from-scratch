from collections import defaultdict, deque
import numpy as np
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld
from common.utils import greedy_probs


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        # 2개 넘어가면 FIFO로 내보내고 2개만 유지...
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        # 일단 경험치를 큐에 넣는데...아직 최소 2개가 안채워졌으면 그냥 나간다..
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        # 이 단계와 다음 단계의 격자위치와 이동방향, 그 사이 보상
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        # 198쪽 식 6.10에 따라 Q 갱신
        next_q = 0 if done else self.Q[next_state, next_action]
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # ε-탐욕화...
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)

            # 목표 지점 바로 전의 격자를 업데이트하려면 목표 지점에서 한 번 더 호출해야 된다...
            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state

    env.render_q(agent.Q)
