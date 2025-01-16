import gymnasium as gym
import time
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
import ale_py

# Carrega/Cria o ambiente (exemplo Pong-ram-v4)
gym.register_envs(ale_py)
env = gym.make('Pong-ram-v4', render_mode='human')
model_path = [
                "logs/best_model.zip",
                "modelos/800ktime_steps.zip",
              "modelos/best_model.zip"
              ]
print(env)
# Carrega o modelo treinado
modelo = DQN.load(model_path[2], env=env)

# Reseta o ambiente
obs, info = env.reset()
terminou = False

# Loop para ver o modelo treinado em ação
while not terminou:
    acao, _ = modelo.predict(obs, deterministic=True)
    obs, recompensa, terminou, truncado, info = env.step(acao)
    env.render()
    time.sleep(0.01)

env.close()