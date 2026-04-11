import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meanings = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.reward_map = np.array([[0, 0, 0, 1], [0, None, 0, -1], [0, 0, 0, 0]])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        # 첫 번째 차원의 길이니까 행의 수 3
        return len(self.reward_map)

    @property
    def width(self):
        # 첫 번째 원소를 꺼내서 보면 두번째 차원의 길이...4
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                # 잠시 멈춤 기능, for 문이 잠시 멈췄다 다시 호출되니 이어서 다음 값 반환...
                yield (h, w)

    def next_state(self, state, action):
        # 이동 순서는 서동북남
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 벽으로 가는 상황이라면 원래 상태 그대로 반환하고
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            # 아니지만 중간에 있는 벽에 들어가도 원래 상태를 반환
            next_state = state

        return next_state

    # 원래 보상은 r(s, a, s')이지만 이번에는 s'만으로 보상
    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        # 다음 위치로 받고
        next_state = self.next_state(self.agent_state, action)
        # 그 위치에 따른 보상, 목표 달성, 이동 정리
        reward = self.reward(self.agent_state, action, next_state)
        done = next_state == self.goal_state
        self.agent_state = next_state
        return next_state, reward, done

    # 이 아래는 도구 함수로 그림 그리는 거라 일단 그냥 배껴넣음
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_q(q, print_value)
