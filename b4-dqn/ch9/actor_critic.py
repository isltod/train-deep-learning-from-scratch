from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F
import gymnasium


class PolicyNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, next_state, reward, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # 상태 가치 함수 V의 손실 계산 - TD Target 계산
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)

        # ..에 이어 정책 pi 손실 계산 - 304쪽 식 9.6
        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        loss_v.backward()
        self.optimizer_v.update()
        self.pi.cleargrads()
        loss_pi.backward()
        self.optimizer_pi.update()


if __name__ == "__main__":
    env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
    agent = Agent()
    reward_history = []
    episodes = 3000

    for episode in tqdm(range(episodes)):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, prob, next_state, reward, done)
            total_reward += reward
            state = next_state

        reward_history.append(total_reward)

    plt.plot(reward_history)
    plt.show()

    env.close()
