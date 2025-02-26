import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time  # para adicionar uma pequena pausa durante o render

# --- Rede Neural (MLP) para classificação ---
class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_actions=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- Função de simulação (rollout) ---
def simulate_rollout(state, action, horizon, env_name='CartPole-v1'):
    env = gym.make(env_name)
    env.reset()
    # Configura o estado atual (para CartPole isso é possível)
    env.env.state = np.copy(state)
    total_reward = 0.0
    done = False
    
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    for _ in range(horizon - 1):
        if done:
            break
        a = env.action_space.sample()  # ação aleatória
        obs, reward, done, _ = env.step(a)
        total_reward += reward

    env.close()
    return total_reward

# --- Função do planejador UCT simples ---
def uct_action(env, n_rollouts=50, horizon=20):
    state = np.copy(env.env.state)
    action_values = {}
    for a in range(env.action_space.n):
        rewards = []
        for _ in range(n_rollouts):
            r = simulate_rollout(state, a, horizon, env.spec.id)
            rewards.append(r)
        action_values[a] = np.mean(rewards)
    best_action = max(action_values, key=action_values.get)
    return best_action

# --- Coleta de dados usando o planejador UCT ---
def collect_data(env, num_episodes=50):
    states = []
    actions = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = uct_action(env, n_rollouts=50, horizon=20)
            states.append(obs)
            actions.append(a)
            obs, reward, done, _ = env.step(a)
        print(f"Episódio {episode+1} finalizado com {len(states)} amostras acumuladas.")
    return np.array(states), np.array(actions)

# --- Treinamento do modelo ---
def train_policy(states, actions, num_epochs=20, batch_size=32, learning_rate=1e-3):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    dataset_size = len(states)
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    
    for epoch in range(num_epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_states = states_tensor[indices]
            batch_actions = actions_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
    return model

# --- Avaliação do modelo treinado (sem render) ---
def evaluate_policy(env, model, num_episodes=10):
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        print(f"Episódio de avaliação {episode+1}: recompensa = {episode_reward}")
    avg_reward = np.mean(rewards)
    print("Recompensa média na avaliação:", avg_reward)
    return rewards

# --- Função para ver o agente jogando com renderização ---
def play_agent(env, model, num_episodes=5, delay=0.02):
    model.eval()  # coloca o modelo em modo de avaliação
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()  # exibe o ambiente
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            time.sleep(delay)  # pausa para melhor visualização
        print(f"Episódio jogado: Recompensa = {total_reward}")
    env.close()

def main():
    env = gym.make('CartPole-v1', render_mode='human')
    
    print("Coletando dados com o planejador UCT...")
    states, actions = collect_data(env, num_episodes=500)
    print(f"Dados coletados: {len(states)} amostras.")
    
    print("Treinando a política (MLP)...")
    model = train_policy(states, actions, num_epochs=20)
    
    # Salvar o modelo treinado
    model_path = "cartpole_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")
    
    print("Avaliando o agente treinado (sem render) ...")
    evaluate_policy(env, model, num_episodes=10)
    
    # Carregar o modelo salvo (opcional, para demonstrar como carregar)
    loaded_model = MLP()
    loaded_model.load_state_dict(torch.load(model_path))
    print("Modelo carregado para jogar.")
    
    print("Mostrando o agente jogando com renderização...")
    play_agent(env, loaded_model, num_episodes=5)
    
if __name__ == '__main__':
    main()
