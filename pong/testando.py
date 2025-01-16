import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py
# Classe personalizada para o extrator de recursos (MLP)
gym.register_envs(ale_py)  # unnecessary but helpful for IDEs




# Wrapper para modificar recompensas
def custom_reward_wrapper(env):
    class CustomRewardEnv(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)

        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            ball_x = obs[49]  # Posição aproximada da bola em RAM
            agent_x = obs[51]  # Posição aproximada do agente em RAM
            
            # Penalidade se a bola passar pelo agente
            if ball_x < agent_x:
                reward += -5

            return obs, reward, done, truncated, info

    return CustomRewardEnv(env)

# Configurando os ambientes com Monitor e wrapper de recompensa
env = Monitor(custom_reward_wrapper(gym.make('Pong-ram-v4')))
eval_env = Monitor(custom_reward_wrapper(gym.make('Pong-ram-v4')))

# Callback personalizado para registrar recompensas
class RewardLoggerCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][-1])
        return super()._on_step()

# Configurando o callback de avaliação com log de recompensas
eval_callback = RewardLoggerCallback(
    eval_env, 
    best_model_save_path='./logs/',
    log_path='./logs/', 
    eval_freq=10000,
    deterministic=True, 
    render=False
)



# Criando o modelo DQN
modelo = DQN(
    policy="MlpPolicy",
    env=env,
    gamma=0.99,
    learning_rate=1e-5,  # Taxa de aprendizado reduzida para maior estabilidade
    buffer_size=500000,  # Buffer de experiência maior
    exploration_fraction=0.05,  # Reduz a exploração ao longo do tempo
    exploration_final_eps=0.01,
    exploration_initial_eps=1.0,
    train_freq=8,  # Treina com frequência menor, mas com mais dados
    batch_size=64,
    learning_starts=50000,

    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# Treinando o modelo
modelo.learn(total_timesteps=1500000, log_interval=10, callback=eval_callback)

# Salvando o modelo treinado
modelo.save("dqn_ram_pong_model_modified")
