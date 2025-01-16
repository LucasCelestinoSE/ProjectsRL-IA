import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
import time
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd


# Cria o ambiente com o wrapper personalizado
env = gym.make('CartPole-v1')


# Cria o modelo DQN com a política MlpPolicy
model = DQN(policy=MlpPolicy,
             env=env,
             gamma=0.1,
             learning_rate=0.0005,
             buffer_size=5000,
             exploration_fraction=0.1,
             exploration_final_eps=0.1,
             exploration_initial_eps=1.0,
             batch_size=32,
             learning_starts=15000,
             verbose=1)

# Treina o modelo
model.learn(total_timesteps=100000, log_interval=20)

# Salva o modelo treinado
model.save("dqn_cartpole_model_low_gamma")

