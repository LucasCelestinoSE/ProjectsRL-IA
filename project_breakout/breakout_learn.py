import gymnasium as gym
import ale_py
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

# Função para criar o ambiente Atari
gym.register_envs(ale_py) #Apenas se necessário

def make_atari_env(env_id, n_envs=1, seed=0):
    def make_env():
        env = gym.make(env_id, render_mode="rgb_array")
        env = AtariWrapper(env)
        return env
    return DummyVecEnv([make_env for _ in range(n_envs)])

# Cria o ambiente BreakoutNoFrameskip-v4
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=2, seed=0)
env = VecFrameStack(env, n_stack=4)
modelo = PPO(policy=CnnPolicy, env=env, gamma=0.99, n_steps=128, learning_rate=0.00025, vf_coef=0.55, verbose=1)

inicio = time.time()
modelo.learn(total_timesteps=40000000, log_interval=10)
fim = time.time()
print("Tempo em horas: ", (fim-inicio))

# Salva o modelo treinado
modelo.save("breakoutmodel")