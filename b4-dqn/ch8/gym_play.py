import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("CartPole-v1", render_mode="human")
state = env.reset()[0]
print("state:", state)

action_space = env.action_space
print("action_space:", action_space)

action = action_space.sample()
print("action:", action)

next_state, reward, terminated, truncated, info = env.step(action)
print("next_state:", next_state)
print("reward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)
print("info:", info)

env.close()

# 이거부터 새로 만들어야 시작된다...
env = gymnasium.make("CartPole-v1", render_mode="human")
state = env.reset()[0]
done = False

next_state = state
while not done:
    env.render()
    # 원래는 엄청 빨리 끝나는데 이걸 넣으면 좀 낫다...막대의 각속도와 반대 방향으로 action을 준다...
    if next_state[3] > 0:
        action = 0
    else:
        action = 1
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print("next_state:", next_state)
env.close()
