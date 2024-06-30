# 實現貪食蛇遊戲的環境和DQN模型

import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

class SnakeGame:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # 初始化遊戲狀態
        self.snake = [(0, 0)]
        self.direction = (0, 1)  # 初始方向向右
        self.place_food()
        self.game_over = False

    def place_food(self):
        # 隨機放置食物
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def step(self, action):
        # 執行動作並返回狀態、獎勵和遊戲是否結束
        new_direction = self.direction
        if action == 0:  # 左轉
            new_direction = (-self.direction[1], self.direction[0])
        elif action == 1:  # 直行
            pass  # 保持原方向
        elif action == 2:  # 右轉
            new_direction = (self.direction[1], -self.direction[0])

        new_head = (self.snake[-1][0] + new_direction[0], self.snake[-1][1] + new_direction[1])

        if (new_head in self.snake[:-1] or
                new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size):
            reward = -1
            self.game_over = True
        else:
            self.snake.append(new_head)
            if new_head == self.food:
                reward = 1
                self.place_food()
            else:
                reward = 0
                self.snake = self.snake[1:]

        self.direction = new_direction

        return self.get_state(), reward, self.game_over

    def get_state(self):
        # 返回當前狀態的表示，這裡使用簡單的矩陣表示
        state = np.zeros((self.grid_size, self.grid_size))
        for i, (x, y) in enumerate(self.snake):
            state[x, y] = 1 if i == len(self.snake) - 1 else -1
        state[self.food] = 0.5
        return state

class DQNSolver:
    def __init__(self, grid_size, action_space):
        self.grid_size = grid_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(self.grid_size, self.grid_size, 1)))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state.reshape(1, self.grid_size, self.grid_size, 1))
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, self.grid_size, self.grid_size, 1))[0])
            target_f = self.model.predict(state.reshape(1, self.grid_size, self.grid_size, 1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, self.grid_size, self.grid_size, 1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 主函數
if __name__ == "__main__":
    grid_size = 10
    action_space = 3  # 左轉、直行、右轉
    episodes = 1000
    batch_size = 32

    snake_game = SnakeGame(grid_size)
    dqn_solver = DQNSolver(grid_size, action_space)

    for episode in range(episodes):
        snake_game.reset()
        state = snake_game.get_state().reshape(grid_size, grid_size, 1)
        total_reward = 0
        while not snake_game.game_over:
            action = dqn_solver.act(state)
            next_state, reward, done = snake_game.step(action)
            next_state = next_state.reshape(grid_size, grid_size, 1)
            dqn_solver.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {episode+1}/{episodes}, Score: {total_reward}")
                break
            dqn_solver.replay(batch_size)
