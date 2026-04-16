import copy
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F
from replay_bufffer import ReplayBuffer
import gymnasium
import matplotlib.pyplot as plt
from tqdm import tqdm


class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    # x는 state, 받으면 linear->relu->linear->relu->linear 처리해서 Q함수 값들로 만들어 반환
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        # 같은 QNet에서 두 개를 만들고
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        # target 말고 qnet만 optimizer에 등록 - qnet만 매번 가중치 갱신, target은 고정 상태로...
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        # 정해진 횟수만큼 반복 후 target을 qnet과 같게 만들기...
        # deepcopy로 안하면 참조를 복사해서 qnet 갱신할 때 target까지 갱신되버린다...
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    # 그러니까 이게 Q 값을 좋게 갱신하는건데...그게 경험 재생으로...배치 만큼 받아와서 한다...
    def update(self, state, action, reward, next_state, done):
        # 우선 현재 받아 본 상태, 행동, 보상, 다음 상태, 도달 여부를 버퍼에 보관하고
        self.replay_buffer.add(state, action, reward, next_state, done)
        # 배치 크기만큼 데이터가 모이지 않았다면 그냥 나간다...
        if len(self.replay_buffer) < self.batch_size:
            return

        # 배치 크기 이상 데이터가 있다면, 랜덤 추출로 배치 크기 만큼 경험을 받아와서...
        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        # 배치x행동 종류 배열로 Q 함수 값 받고, 그 중 현재 행동에 해당하는 Q값만 추리고...
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        # 이건 매번 갱신되지는 않는 target Q, 거기서 배치x행동 종류 배열로 Q값 받고,
        next_qs = self.qnet_target(next_state)
        # 그 중 max 행동 값이 정답지...배치x최대행동Q
        max_next_q = next_qs.data.max(axis=1)
        # (1-done)은 끝이면 0, 남았으면 1, q를 종료에서 0으로 만드는 효과...
        target = reward + (1 - done) * self.gamma * max_next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        # 가중치 갱신은 qnet만 한다...
        self.optimizer.update()


if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    env = gymnasium.make("CartPole-v1", render_mode="human")
    agent = DQNAgent()
    reward_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            # 실제 게임은 여기서 하는거고...에이전트한테 action 받아서 환경한테 step 시키는거...
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            # 여기서 배운 교훈을 저장하고, 그 중에 랜덤 선택해서 경험 재생으로 전략을 수정하는게 이거고...
            agent.update(state, action, reward, next_state, done)
            state = next_state

        reward_history.append(episode_reward)

        if episode % sync_interval == 0:
            agent.sync_qnet()

    # 에피소드 별 보상 합 추이
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(reward_history)
    plt.show()

    # 학습 된 에이전트를 탐욕 행동을 선택하도록 해서 플레이...
    agent.epsilon = 0
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    print(total_reward)
    env.close()
