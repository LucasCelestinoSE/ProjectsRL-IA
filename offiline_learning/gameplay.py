import gymnasium as gym
import numpy as np
import cv2
import time
from stable_baselines3 import DQN
import ale_py
import math
import torch

class PongPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, 84, 84),
            dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, axis=0)

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = list(range(3))

class MCTSPlayer:
    def __init__(self, env, model_path, num_simulations=20, exploration_weight=1.0):
        self.env = env
        self.dqn = DQN.load(model_path)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
    
    def get_action(self, state):
        root = Node(state)
        
        # Executa as simulações MCTS
        for _ in range(self.num_simulations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        
        return self.get_best_action(root)
    
    def select(self, node):
        while node.untried_actions == [] and node.children:
            node = self.get_best_child(node)
        
        if node.untried_actions:
            # Usa DQN para guiar a seleção
            q_values = self.dqn.q_net(torch.tensor(np.expand_dims(node.state, axis=0), dtype=torch.float32))
            action_probs = self.softmax(q_values.detach().numpy()[0])
            
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
        state = node.state.copy()
        done = False
        total_reward = 0
        max_steps = 50
        
        for _ in range(max_steps):
            if done:
                break
            
            # Usa o DQN para guiar a simulação
            action, _ = self.dqn.predict(state, deterministic=False)
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
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

def main():
    # Registra os ambientes do ALE
    gym.register_envs(ale_py)
    
    # Cria e prepara o ambiente
    env = gym.make('Pong-v4', render_mode='human')
    env = PongPreprocessing(env)
    
    # Cria o jogador MCTS-DQN
    player = MCTSPlayer(env, "pong_dqn.zip", num_simulations=20)
    
    # Loop principal do jogo
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Obtém a ação usando MCTS guiado pelo DQN
        action = player.get_action(obs)
        
        # Executa a ação
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        # Pequena pausa para visualização
        time.sleep(0.01)
        
    print(f"Jogo finalizado! Recompensa total: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()