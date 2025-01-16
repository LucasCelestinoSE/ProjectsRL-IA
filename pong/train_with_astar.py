import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import heapq

# import aly_py se necessário
# import aly_py

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.net(obs)

class PositiveRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        shaped_reward = reward
        if shaped_reward < 0:
            shaped_reward *= 0.5  # Reduz a penalidade
        return shaped_reward

class AStarPlanner:
    def __init__(self, env):
        self.env = env

    def heuristic(self, start, goal):
        # Distância Manhattan entre a bola e a barra do agente
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def get_neighbors(self, pos):
        # Movimentos possíveis: cima, baixo, parado
        return [(pos[0], pos[1] - 1), (pos[0], pos[1] + 1), pos]

    def a_star_search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1  # Assume custo de movimento constante

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def plan(self, start, goal):
        return self.a_star_search(start, goal)

class CombinedAgent:
    def __init__(self, dqn_model, env):
        self.dqn_model = dqn_model
        self.env = env
        self.planner = AStarPlanner(env)

    def act(self, obs):
        # Extrair posições da bola e da barra do agente da RAM
        ball_pos = (obs[49], obs[50])  # Exemplo de índices da RAM
        agent_pos = (obs[51], obs[52])  # Exemplo de índices da RAM

        # Planejar caminho usando A*
        path = self.planner.plan(agent_pos, ball_pos)
        if path:
            next_pos = path[0]
            if next_pos[1] < agent_pos[1]:
                action = 2  # Mover para cima
            elif next_pos[1] > agent_pos[1]:
                action = 3  # Mover para baixo
            else:
                action = 0  # Ficar parado
        else:
            action, _ = self.dqn_model.predict(obs, deterministic=True)

        return action

# Crie o ambiente base e aplique o wrapper
base_env = gym.make('Pong-ram-v4')
wrapped_env = PositiveRewardWrapper(base_env)
env = DummyVecEnv([lambda: wrapped_env])

eval_env = gym.make('Pong-ram-v4')
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10000,
                             deterministic=True, render=False)

policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=256),
)

dqn_model = DQN(
    policy="MlpPolicy",
    env=env,
    gamma=0.99,
    learning_rate=5e-5,
    buffer_size=200000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    exploration_initial_eps=1.0,
    train_freq=4,
    batch_size=64,
    learning_starts=50000,
    policy_kwargs=policy_kwargs,
    verbose=1
)

dqn_model.learn(total_timesteps=5000000, log_interval=10, callback=eval_callback)
dqn_model.save("dqn_ram_pong_model")

# Use CombinedAgent for acting in the environment
combined_agent = CombinedAgent(dqn_model, env)
obs = env.reset()
done = False
while not done:
    action = combined_agent.act(obs)
    obs, reward, done, info = env.step(action)