import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque

# --- Definição da Rede Neural para o DQN ---
class DQN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_actions=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- Replay Buffer Prioritário ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("O buffer está vazio!")
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)
    
    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# --- Simulação de Rollout com política opcional ---
def simulate_rollout(state, action, horizon, env_name='CartPole-v1', policy=None):
    env = gym.make(env_name)
    env.reset()
    # Atribui o estado desejado (note que nem todas as versões do Gym permitem isso)
    env.env.state = np.copy(state)
    total_reward = 0.0
    done = False
    
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    for _ in range(horizon - 1):
        if done:
            break
        if policy is not None:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
            a = torch.argmax(logits, dim=1).item()
        else:
            a = env.action_space.sample()
        obs, reward, done, _ = env.step(a)
        total_reward += reward

    env.close()
    return total_reward

# --- Planejamento UCT com rollouts opcionais ---
def uct_action(env, n_rollouts=20, horizon=50, rollout_policy=None):
    state = np.copy(env.env.state)
    action_values = {}
    for a in range(env.action_space.n):
        rewards = []
        for _ in range(n_rollouts):
            r = simulate_rollout(state, a, horizon, env.spec.id, policy=rollout_policy)
            rewards.append(r)
        action_values[a] = np.mean(rewards)
    return max(action_values, key=action_values.get)

# --- Coleta de dados com UCT (aceita política para rollout) ---
def collect_data(env, buffer, num_episodes=500, rollout_policy=None):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = uct_action(env, n_rollouts=50, horizon=20, rollout_policy=rollout_policy)
            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
        print(f"Episódio {episode+1} coletado, buffer tamanho: {len(buffer)}")

# --- Treinamento do DQN Offline com PER ---
def train_dqn(dqn, buffer, num_epochs=0, batch_size=128, learning_rate=1e-3, gamma=0.99):
    dqn.train()
    criterion = nn.MSELoss(reduction='none')  # perda por amostra
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        samples, indices, weights = buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q(s, a) para as ações tomadas
        q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Q target: Bellman update usando o valor máximo da próxima ação
        next_q_values = dqn(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * gamma * next_q_values

        losses = criterion(q_values, expected_q_values)
        loss = (losses * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Atualiza as prioridades com base no erro TD (loss por amostra)
        new_priorities = losses.detach().cpu().numpy() + 1e-6
        buffer.update_priorities(indices, new_priorities)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

# --- Avaliação do modelo treinado ---
def evaluate_policy(env, model, num_episodes=10):
    model.eval()
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        print(f"Episódio de avaliação {episode+1}: recompensa = {episode_reward}")
    print("Recompensa média:", np.mean(rewards))
    return rewards

# --- Função para ver o DQN jogando ---
def play_agent(env, model, num_episodes=5, delay=0.02):
    model.eval()
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            time.sleep(delay)
        print(f"Episódio jogado: Recompensa = {total_reward}")
    env.close()

# --- Função principal ---
def main():
    env = gym.make('CartPole-v1')
    
    # Instancia o buffer com experiência prioritária
    buffer = PrioritizedReplayBuffer(capacity=100000)
    
    # Fase 1: Coleta de dados com UCT utilizando rollouts aleatórios
    print("Coletando dados com o planejador UCT (rollouts aleatórios)...")
    collect_data(env, buffer, num_episodes=200, rollout_policy=None)
    
    # Treina o DQN inicialmente
    dqn = DQN()
    print("Treinando DQN offline (fase 1)...")
    train_dqn(dqn, buffer, num_epochs=10)
    
    # Fase 2: Coleta de dados com UCT utilizando a política DQN para guiar os rollouts
    print("Coletando dados com UCT utilizando política DQN (rollouts guiados)...")
    collect_data(env, buffer, num_episodes=300, rollout_policy=dqn)
    
    # Treinamento final do DQN
    print("Treinando DQN offline (fase 2)...")
    train_dqn(dqn, buffer, num_epochs=20)
    
    # Salvando e carregando o modelo
    torch.save(dqn.state_dict(), "dqn_cartpole.pth")
    print("Modelo salvo!")
    
    dqn.load_state_dict(torch.load("dqn_cartpole.pth"))
    print("Modelo carregado para jogar.")
    
    print("Avaliando DQN...")
    evaluate_policy(env, dqn, num_episodes=10)
    
    print("Mostrando DQN jogando...")
    play_agent(env, dqn, num_episodes=5)
    
    env.close()

if __name__ == '__main__':
    main()
