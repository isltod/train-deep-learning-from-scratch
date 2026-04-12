from collections import defaultdict
import numpy as np
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld


class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        # pi = {(0,0): {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, ...} 같은 모양이되고...
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        # 행동을 시키면 생기는 상태, 행동, 보상 - 경험을 저장
        self.memory = []

    def get_action(self, state):
        # 뭔가 따로 설정된 것이 없다면 {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}를 반환
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        # 확률에 따라 0,1,2,3 중에 하나를 랜덤 선택
        action = np.random.choice(actions, p=probs)
        return action

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        # 역방향으로 따라가야 162쪽 식이 된다...
        for state, action, reward in reversed(self.memory):
            # 이게 162쪽 식을 한 줄로 쓴거고...
            G = reward + self.gamma * G
            self.cnts[state] += 1
            # 157쪽 식 5.2를 증분식으로 쓴 거...
            self.V[state] += (G - self.V[state]) / self.cnts[state]


if __name__ == "__main__":
    env = GridWorld()
    agent = RandomAgent()

    # 반복을 10,000 정도 하면 정답과 거의 유사해진다...
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            # 서동북남 중 하나를 확률적으로 얻고,
            action = agent.get_action(state)
            # 그리로 이동해서 다음 위치, 보상, 종료 여부를 얻고
            next_state, reward, done = env.step(action)
            # 경험을 저장하고
            agent.add(state, action, reward)
            if done:
                agent.eval()
                break
            state = next_state

    env.render_v(agent.V)
