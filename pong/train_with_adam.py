import tensorflow as tf
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import ale_py

# Carrega/Cria o ambiente (exemplo Pong-ram-v4)
gym.register_envs(ale_py)

class PositiveRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward + 1  # Adjust this logic as needed

class CustomMLP(tf.keras.Model):
    def __init__(self, observation_space, features_dim=256):
        super(CustomMLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(features_dim, activation='relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

class CustomPolicy(DQN):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs)
        self.model = CustomMLP(self.observation_space, features_dim=256)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    def train(self, gradient_steps, batch_size=64):
        for _ in range(gradient_steps):
            # Sample a batch of experiences from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size)
            with tf.GradientTape() as tape:
                # Compute the loss
                loss = self.compute_loss(replay_data)
            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)
            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def compute_loss(self, replay_data):
        # Implement the loss computation
        pass

# Crie o ambiente base e aplique o wrapper
base_env = gym.make('Pong-ram-v4')
wrapped_env = PositiveRewardWrapper(base_env)
env = DummyVecEnv([lambda: wrapped_env])

eval_env = gym.make('Pong-ram-v4')
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10000,
                             deterministic=True, render=False)

policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=256),
)

dqn_model = CustomPolicy(
    policy="MlpPolicy",
    env=env,
    gamma=0.99,
    learning_rate=5e-5,
    buffer_size=200000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    exploration_initial_eps=1.0,
    train_freq=4,
    batch_size=64,
    learning_starts=50000,
    policy_kwargs=policy_kwargs,
    verbose=1
)

dqn_model.learn(total_timesteps=5000000, log_interval=10, callback=eval_callback)
dqn_model.save("dqn_ram_pong_model")