import gymnasium as gym
import numpy as np
from collections import defaultdict
import math
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import ale_py

gym.register_envs(ale_py)

class RamPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # A RAM do Atari tem 128 bytes
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1.0,
            shape=(128,),
            dtype=np.float32
        )

    def observation(self, obs):
        # Normaliza os valores da RAM entre 0 e 1
        return obs.astype(np.float32) / 255.0

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = list(range(3))

class MCTSMLPPong:
    def __init__(self, env, model_path=None, exploration_weight=1.0):
        self.env = env
        self.exploration_weight = exploration_weight
        
        # Configura o DQN com política MLP
        policy_kwargs = dict(
            net_arch=[128, 64]  # Camadas menores para input menor
        )
        
        if model_path:
            self.dqn = DQN.load(model_path)
        else:
            self.dqn = DQN(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=0.0001,
                buffer_size=500000 ,
                learning_starts=1000,
                batch_size=64,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                verbose=1
            )
        
    def get_dqn_action_probs(self, state):
        """Obtém as probabilidades de ação do DQN para um dado estado"""
        state = np.array(state, dtype=np.float32)
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
            
        with torch.no_grad():
            q_values = self.dqn.q_net(torch.as_tensor(state))
            q_values = q_values.detach().numpy()
        
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        
        return probs[0]
    
    def choose_action(self, state, num_simulations=50):
        root = Node(state)
        
        for _ in range(num_simulations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        
        return self.get_best_action(root)
    
    def select(self, node):
        while node.untried_actions == [] and node.children:
            node = self.get_best_child(node)
        
        if node.untried_actions:
            action_probs = self.get_dqn_action_probs(node.state)
            available_probs = [action_probs[a] for a in node.untried_actions]
            available_probs = available_probs / np.sum(available_probs)
            
            action = np.random.choice(node.untried_actions, p=available_probs)
            node.untried_actions.remove(action)
            
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            child = Node(next_state, parent=node)
            node.children[action] = child
            return child
        
        return node
    
    def get_best_child(self, node):
        def ucb1(n, child):
            exploitation = child.value / child.visits
            exploration = math.sqrt(2 * math.log(n.visits) / child.visits)
            return exploitation + self.exploration_weight * exploration
        
        return max(
            node.children.items(),
            key=lambda item: ucb1(node, item[1])
        )[1]
    
    def simulate(self, node):
        state = node.state
        done = False
        total_reward = 0
        max_steps = 50
        
        for _ in range(max_steps):
            if done:
                break
            
            action_probs = self.get_dqn_action_probs(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            
            state, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            
        return total_reward
    
    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_best_action(self, node):
        return max(
            node.children.items(),
            key=lambda item: item[1].visits
        )[0]
    
    def train_dqn(self, total_timesteps=1000000):
        self.dqn.learn(total_timesteps=total_timesteps)
    
    def save_dqn(self, path):
        self.dqn.save(path)

def main():
    # Cria o ambiente com pré-processamento
    env = gym.make("Pong-ram-v4", render_mode="human")
    env = RamPreprocessing(env)
    
    # Cria o agente híbrido com MLP
    agent = MCTSMLPPong(env)
    
    # Treina o DQN
    print("Treinando DQN com MLP...")
    agent.train_dqn()
    
    # Salva o modelo treinado
    agent.save_dqn("pong_mlp_dqn")
    
    # Executa episódios usando a combinação MCTS-MLP
    max_episodes = 5
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
        print(f"Episódio {episode + 1}: Recompensa total = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    main()