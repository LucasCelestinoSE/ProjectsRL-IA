import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy
import time
env_id = "BipedalWalker-v3"



def cria_env(nome_env, num,seed= 0):
    def _init():
        env = gym.make(nome_env)
        return env
    return _init


num_cpu = 16
CheckpointCallback = CheckpointCallback(save_freq=20833,save_path="logs")
if __name__ == '__main__':
    env = SubprocVecEnv([cria_env(nome_env='BipedalWalker-v3', num=i) for i in range(num_cpu)])
    inicio = time.time()
    modelo = PPO(policy=MlpPolicy,env=env,gamma=0.99, n_steps=5, vf_coef=0.25, learning_rate=0.0007, verbose=1)
    modelo.learn(total_timesteps=20000000, log_interval=100, callback=CheckpointCallback)
    modelo.save("bipewalker_train_teste")
    fim = time.time()
    print("Tempo em horas : ", (fim-inicio) / 3600)

    