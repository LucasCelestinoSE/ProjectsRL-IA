from stable_baselines3 import PPO
from stable_baselines3 import A2C
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
log_path = "Bipedalwalker/logs"
module_path = [log_path + "/rl_model_333328_steps.zip",
                log_path +  "/rl_model_1333312_steps.zip",
                log_path +  "/rl_model_3999936_steps.zip",
                log_path +  "/rl_model_6333232_steps.zip",
                log_path +  "/rl_model_19999680_steps.zip",
               "Bipedalwalker/modelos/a2c-BipedalWalker-v3.zip",
               ]
env = DummyVecEnv([make_env])
# A2C
# Escolha um modelo no array para carregar
modelo = A2C.load(module_path[5])
# PPO
##modelo = PPO.load(module_path[4])
# Reseta o ambiente
obs = env.reset()
terminou = False

# Loop para ver o modelo treinado em ação
while not terminou:
    acao, _ = modelo.predict(obs, deterministic=True)
    obs, reward, terminou, info = env.step(acao)
    env.render()