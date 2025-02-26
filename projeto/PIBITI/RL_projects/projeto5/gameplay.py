from stable_baselines3 import PPO
import gymnasium as gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gym.make("BipedalWalker-v3", render_mode="human")

env = DummyVecEnv([make_env])
modelo = PPO.load("logs/rl_model_6145735_steps.zip", env=env)

# Reseta o ambiente
obs = env.reset()
terminou = False

# Loop para ver o modelo treinado em ação
while not terminou:
    acao, _ = modelo.predict(obs, deterministic=True)
    obs, recompensa, terminou, info = env.step(acao)
    env.render()
    time.sleep(0.01)