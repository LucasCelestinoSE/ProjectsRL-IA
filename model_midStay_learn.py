import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
import time
 # Algumas politicas legais XD XD
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def reward(self, reward):
        # Aumenta a recompensa se o carrinho estiver mais próximo do centro
        x, _, _, _ = self.env.unwrapped.state  # Acesse o ambiente subjacente
        center_bonus = 1.0 - abs(x)  # Quanto mais próximo do centro, maior o bônus
        return reward + center_bonus

# Cria o ambiente com o wrapper personalizado
env = gym.make('CartPole-v1')
env = CustomRewardWrapper(env)

# Cria o modelo DQN com a política MlpPolicy
model = DQN(policy=MlpPolicy,
             env=env,
             gamma=0.99,
             learning_rate=0.0005,
             buffer_size=5000,
             exploration_fraction=0.1,
             exploration_final_eps=0.1,
             exploration_initial_eps=1.0,
             batch_size=32,
             learning_starts=15000,
             verbose=1)

# Treina o modelo
model.learn(total_timesteps=600000, log_interval=20)

# Salva o modelo treinado
model.save("dqn_cartpole_midStay_model")