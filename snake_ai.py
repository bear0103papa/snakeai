import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import time
import os

class SnakeGame:
    def __init__(self, width=15, height=15):  # 減小遊戲區域
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.food = self.generate_food()
        self.score = 0
        self.steps = 0
        self.game_over = False

    def generate_food(self):
        while True:
            food = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if food not in self.snake:
                return food

    def step(self, action):
        self.steps += 1
        head = self.snake[0]
        if action == 0:  # 上
            new_head = (head[0], (head[1] - 1) % self.height)
        elif action == 1:  # 右
            new_head = ((head[0] + 1) % self.width, head[1])
        elif action == 2:  # 下
            new_head = (head[0], (head[1] + 1) % self.height)
        else:  # 左
            new_head = ((head[0] - 1) % self.width, head[1])

        if new_head in self.snake or self.steps > 10000000:  # 添加最大步數限制
            self.game_over = True
            return -10

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
            reward = 10
        else:
            self.snake.pop()
            distance = np.sqrt((new_head[0] - self.food[0])**2 + (new_head[1] - self.food[1])**2)
            reward = 1 / (distance + 1)

        if len(self.snake) >= self.width * self.height * 0.5:  # 如果蛇佔據了一半的遊戲區域，就結束遊戲
            self.game_over = True
            reward += 100  # 給予額外獎勵

        return reward

    def get_state(self):
        state = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i, (x, y) in enumerate(self.snake):
            if i == 0:  # 蛇頭
                state[y, x, 2] = 1  # 藍色通道表示蛇頭
            else:  # 蛇身
                state[y, x, 1] = 1  # 綠色通道表示蛇身
        fx, fy = self.food
        state[fy, fx, 0] = 1  # 紅色通道表示食物
        return state

def create_model(input_shape, num_actions):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(num_actions, activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

def train_model(model, game, episodes=10000, save_interval=100):
    epsilon = 1.0  # 探索率
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    memory = []

    start_time = time.time()
    total_steps = 0

    for episode in range(episodes):
        state = game.get_state()
        total_reward = 0
        game.reset()
        snake_positions = []

        while not game.game_over:
            total_steps += 1
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                q_values = model.predict(np.expand_dims(state, axis=0))[0]
                action = np.argmax(q_values)

            reward = game.step(action)
            next_state = game.get_state()
            total_reward += reward

            snake_positions.append(game.snake[0])  # 記錄蛇頭位置

            memory.append((state, action, reward, next_state, game.game_over))
            state = next_state

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states = np.array([x[0] for x in batch])
                actions = np.array([x[1] for x in batch])
                rewards = np.array([x[2] for x in batch])
                next_states = np.array([x[3] for x in batch])
                dones = np.array([x[4] for x in batch])

                targets = rewards + 0.95 * np.max(model.predict(next_states), axis=1) * (1 - dones)
                target_f = model.predict(states)
                for i, action in enumerate(actions):
                    target_f[i][action] = targets[i]

                model.fit(states, target_f, epochs=1, verbose=0)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        current_time = time.time()
        elapsed_time = current_time - start_time
        steps_per_second = total_steps / elapsed_time
        remaining_episodes = episodes - episode - 1
        estimated_remaining_time = remaining_episodes * (elapsed_time / (episode + 1))

        print(f"Episode {episode + 1}/{episodes}, Score: {game.score}, Total Reward: {total_reward:.2f}")
        print(f"Epsilon: {epsilon:.4f}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Estimated Remaining Time: {estimated_remaining_time:.2f} seconds")
        print(f"Estimated Remaining Episodes: {remaining_episodes}")
        print("--------------------")

        if episode % 100 == 0:
            # 繪製遊戲狀態和蛇的軌跡
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(game.get_state())
            
            # 繪製蛇的軌跡
            for i, (x, y) in enumerate(snake_positions):
                color = plt.cm.viridis(i / len(snake_positions))  # 使用顏色漸變
                ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, fill=True, color=color, alpha=0.5))

            ax.set_title(f"Episode {episode + 1}/{episodes}\nScore: {game.score}, Total Reward: {total_reward:.2f}")
            plt.savefig(f"game_state_episode_{episode + 1}.png", dpi=100)
            plt.close(fig)

        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            model.save(f'snake_ai_model_episode_{episode + 1}.h5')

    return model

# 主程序
game = SnakeGame(15, 15)  # 使用較小的遊戲區域
model = create_model((15, 15, 3), 4)

# 如果存在之前保存的模型，則載入
if os.path.exists('snake_ai_model_latest.h5'):
    model = keras.models.load_model('snake_ai_model_latest.h5')
    print("Loaded existing model.")

trained_model = train_model(model, game, episodes=100, save_interval=1)

# 保存最終模型
trained_model.save('snake_ai_model_final.h5')
