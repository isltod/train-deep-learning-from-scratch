import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        # 엡실론 탐욕 정책, 탐색 확률
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        # 엡실론보다 적은 확률에서는 무작위 탐색을 하고...
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        # 그 외 확률에선 현재 승률 좋은 놈에게 집착한다.
        return np.argmax(self.Qs)


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        agent = Agent(epsilon)
        bandit = Bandit()
        total_reward = 0
        rates = []

        for step in range(steps):
            # 탐색할지 탐욕할지, 슬롯머신 번호 받고
            action = agent.get_action()
            # 슬롯머신 땡겨서 결과 받고
            reward = bandit.play(action)
            # 해당 슬롯머신 시행횟수 1 증가, 보상 기댓값 업데이트 -> 이게 결국 학습...
            agent.update(action, reward)

            # 결과는 머신별로 보는게 아니라 전체적으로...왜냐면 결국 얼마나 딸 수 있느냐가 중요하니까...
            total_reward += reward
            # 그에 따른 승률 기댓값
            rates.append(total_reward / (step + 1))

        # run 별로 승률 기댓값 변화를 넣고
        all_rates[run] = rates

    # 스텝별로 다시 평균 - 200 run의 평균적인 승률 기댓값 변화를 0~999 스텝으로...
    avg_rates = np.average(all_rates, axis=0)

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(avg_rates)
    plt.show()
