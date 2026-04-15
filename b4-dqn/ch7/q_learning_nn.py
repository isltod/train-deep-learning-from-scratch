import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Model
from dezero import optimizers
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("b4-dqn")

from common.gridworld import GridWorld


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)

    # 좌표를 y, x 즉 행, 열로 받아서 위치를 순차로 바꿔서 원핫 인코딩...
    y, x = state
    vec[y * WIDTH + x] = 1
    # 배치 처리를 위해 newaxis 추가해서 기존 데이터를 다 넣는다...1x12 모양이 되겠지...
    return vec[np.newaxis, :]


state = (2, 0)
x = one_hot(state)
print(x)
print(x.shape)


# 이게 앞의 상태(격자) 별 Q 함수와 같은 역할을 하는데...다른 건 데이터를 갱신하고 저장해두는게 아니라는 점...
class QNet(Model):
    def __init__(self):
        super().__init__()
        # 있는 거라곤 Linear 계층 2개 - 거기에 있는 W1, W2, b1, b2(nobias 기본 값이 False이므로 생긴다...)
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(4)

    # Q 값을 요청하면 forward가 처리하는데...
    def forward(self, x):
        # 저장해놓은 Q 값을 주는게 아니라 상태 벡터를 affine -> relu -> affine 해서 (1,4) 벡터를 준다...
        # 이게 해당 격자의 4가지 가능한 행동에 대한 Q 값이다...
        # 앞에서 현재 격자의 Q 값을 다음 격자의 Q 값 등을 이용해서 갱신하던 것을,
        # 여기서는 두 격자의 Q 값을 같게 만들도록 backward(상속)에서 W, b를 수정한다...
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y


qnet = QNet()
state = (2, 0)
state = one_hot(state)

# qnet에 넣으면 Function 클래스 때문에 바로 forward로 연결...
qs = qnet(state)
# 초기 가중치나 편향 등으로 아무튼 (1,4) 벡터 만들어 뱉기...
print(qs)
print(qs.shape)


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        # 249쪽 α에 대한 메모처럼, α = lr 이 된다...
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        # Q는 없고 QNet이라...
        # 어차피 update 식이 다 하는거고, 저장소야 그냥 상태와 행동을 키로 그 값을 저장만 하니...Q건 QNet이건 상관이 없나?
        # Q는 저장만 하고, QNet은 forward를 하는데...
        self.qnet = QNet()
        # self.optimizer = optimizers.SGD(self.lr).setup(self.qnet)
        # 그나마 Adam이 결과가 좀 나은 듯...
        self.optimizer = optimizers.Adam().setup(self.qnet)
        # 왠일인지 AdaGrad는 결과가 아주 않좋다...
        # self.optimizer = optimizers.AdaGrad().setup(self.qnet)

    def get_action(self, state):
        # 6장 q_learning_simple.py 처럼...ε 확률로는 무작위 탐색하고
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # 1-ε 확률로는 탐욕(최대화 결정론적) 행동
            qs = self.qnet(state)
            # 여기서 argmax 뽑는건 역전파 안하나? data는 np.array 아니면 cp.array인데...
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            # 목표 상태는 다음 상태가 없고, next_q는 0, 이건 모양이 [0.0]...
            next_q = np.zeros(1)
        else:
            # linear -> relu -> linear 거쳐서 나온 [q0, q1, q2, q3]
            next_qs = self.qnet(next_state)
            # next_qs는 QNet에서 뱉어낸거니까 Variable 클래스이고...거기 max 연산자 사용...
            next_q = next_qs.max(axis=1)
            # next_q는 역전파에서 제외한다...왜? 이번 회차의 정답이라서 이건 안바꾸나?
            next_q.unchain()
        # 이게 목표 T이고 정답지...
        target = reward + self.gamma * next_q

        # 그에 대비해서 문제인 현재 상태에서의 모든 q값 계산...
        qs = self.qnet(state)
        # 거기서 현재 행동에 대한 q 값들...
        q = qs[:, action]
        # 목표와 q의 오차
        loss = F.mean_squared_error(q, target)

        # 역전파로 QNet의 linear에 있는 W, b 갱신
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        # 이건 그냥 그래프 그릴려고 손실값 저장하는 거니까 큰 의미 없고...
        return loss.data


if __name__ == "__main__":
    agent = QLearningAgent()
    env = GridWorld()

    episodes = 1000
    loss_history = []
    # 목표 지점까지 가는 에피소드를 천 번 반복하면서...
    for episode in range(episodes):
        # 매번 에이전트의 위치를 왼쪽 아래(원핫으로)로 보내고...
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0, 0
        done = False

        # 목표 지점에 도달할 때까지...
        while not done:
            # 현재 위치에서 다음 방향을 받고
            action = agent.get_action(state)
            # 그래서 간 위치, 거기의 보상, 종료 여부를 받고
            next_state, reward, done = env.step(action)
            # 위치만 원핫으로...
            next_state = one_hot(next_state)

            # 여기서는 update에 넘기고, 손실만 기록하는 걸로...실제 가중치 갱신은 update에서...
            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

        loss_history.append(total_loss / cnt)

    plt.plot(loss_history)
    plt.show()

    # 상태, 행동 별 Q 함수 값을 그려보자...
    Q = {}
    for state in env.states():
        # 상태별 q 함수 값들을 한 번에 받아놓고...(1,4) 형태...
        qs = agent.qnet(one_hot(state))
        for action in env.actions():
            # 그 중 행동별 q 함수 값을 추려서 Q[상태, 행동] 키로 넣고 그린다..
            q = qs[:, action]
            Q[state, action] = float(q.data.item())

    env.render_q(Q)
