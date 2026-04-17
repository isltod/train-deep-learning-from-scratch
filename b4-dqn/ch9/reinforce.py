from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F
import gymnasium


class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2
        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        # 이건 행렬곱 시키려고 그냥 차원 하나 추가하고
        state = state[np.newaxis, :]
        probs = self.pi(state)
        # 그렇게 해서 나와봤자 1개가 새 축으로 감싸졌으니 0번을 쓰면 되고...
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        self.memory.append((reward, prob))

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        # 162쪽 식에 의한 수익 기댓값...
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            # 단순 정책경사법은 G total을 구해 그걸 -F.log(prob)에 곱했는데, 여기서는 해당 단계의 G 까지만 곱한다...
            loss += -F.log(prob) * G

        # 그걸 역전파로 미분시켜서 매개변수 갱신
        loss.backward()
        self.optimizer.update()
        self.memory = []


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

            agent.add(reward, prob)
            total_reward += reward
            state = next_state

        agent.update()
        reward_history.append(total_reward)

    plt.plot(reward_history)
    plt.show()

    env.close()
