import gymnasium as gym
import torch
import ale_py
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Classe personalizada para o extrator de recursos (MLP)
gym.register_envs(ale_py)  # unnecessary but helpful for IDEs
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

# Função para plotar recompensas ao longo do treinamento
def plot_rewards(rewards):
    plt.plot(np.cumsum(rewards))
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa Total ao Longo do Treinamento')
    plt.show()

# Configurando o ambiente com Monitor para salvar logs
env = Monitor(gym.make('Pong-ram-v4'))
eval_env = Monitor(gym.make('Pong-ram-v4'))

# Callback personalizado para registrar recompensas
class RewardLoggerCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards = []

    def _on_step(self) -> bool:
        # Registra a recompensa acumulada no episódio
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

# Configurações da política com a rede personalizada
policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=256),
)


# Carregando o modelo pré-treinado
modelo = DQN.load("800ktime_steps", env=env)

# Continuando o treinamento com o modelo carregado
modelo.learn(total_timesteps=10000000, log_interval=10, callback=eval_callback)

# Salvando novamente o modelo após o treinamento adicional
modelo.save("dqn_ram_pong_model_continuado")

# Avaliando o modelo após o treinamento adicional
mean_reward, std_reward = evaluate_policy(modelo, eval_env, n_eval_episodes=10, render=False)
print(f"Média da recompensa após continuação: {mean_reward} +/- {std_reward}")