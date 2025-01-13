import gymnasium as gym
from stable_baselines3 import DQN
import time
import random
import keyboard
import sys
from gymnasium.wrappers import RecordVideo
envId = "CartPole-v1"
model_path ="dqn_cartpole_midStay_model.zip"
# Cria o ambiente
env = gym.make(envId, render_mode="human")  



"""Caso queira gravar um vídeo"""
#env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda e: True)
# Coloque o parametro render_mode = " rgb_array"

# Carrega o modelo treinado
model = DQN.load(model_path)
# Reseta o ambiente
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]  # Pegue apenas a observação
terminou = False
recompensa_total = 0

# Loop para ver o modelo treinado em ação
while not terminou:
    acao, _ = model.predict(obs, deterministic=True)
    
    # Executa a ação no ambiente
    obs, recompensa, terminou, truncado, info = env.step(acao)
    
    # Acumula a recompensa total
    recompensa_total += recompensa
    if recompensa_total > 500:
        break
    # Renderiza o ambiente
    env.render()
    
    # Adiciona um pequeno atraso para visualizar melhor
    time.sleep(0.01)
    

# Fecha o ambiente
env.close()

# Imprime a recompensa total
print("Recompensa Total:", recompensa_total)

