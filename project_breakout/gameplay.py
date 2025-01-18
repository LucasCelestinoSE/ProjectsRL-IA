import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Register ALE environments if necessary
gym.register_envs(ale_py)

# Create and wrap the environment
env = gym.make('BreakoutNoFrameskip-v4', render_mode="human")
env = AtariWrapper(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# Load the model
checkpoint = ["project_breakout/breakoutmodel.zip", "project_breakout/ppo-BreakoutNoFrameskip-v4.zip"]

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}
model = PPO.load(checkpoint[1], custom_objects=custom_objects)

# Reset the environment
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()