from collections import deque
import numpy as np
import random
import gymnasium


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        # 무작위 선택해서 각각 나눠서 반환
        batch = random.sample(self.buffer, self.batch_size)
        # 이 코드는 책과는 다른데, 이게 더 효율적일 것 같다는 생각이...
        state, action, reward, next_state, done = map(np.asarray, zip(*batch))
        return state, action, reward, next_state, done


if __name__ == "__main__":
    env = gymnasium.make("CartPole-v1")
    replay_buffer = ReplayBuffer(capacity=10000, batch_size=32)

    for episode in range(10):
        state = env.reset()[0]
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

    print(len(replay_buffer))
    state, action, reward, next_state, done = replay_buffer.get_batch()
    print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)

    env.close()
