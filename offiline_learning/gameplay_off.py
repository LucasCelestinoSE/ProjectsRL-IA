import gym
import torch
import time

# Supondo que a classe DQN (ou MLP) esteja definida, por exemplo:
class DQN(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_actions=2):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Função para ver o agente jogando com renderização
def play_agent(env, model, num_episodes=5, delay=0.02):
    model.eval()  # Modo avaliação
    for episode in range(num_episodes):
        obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()  # Renderiza o ambiente
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            result = env.step(action)
            # Compatibilidade com diferentes versões do Gym:
            if len(result) == 5:
                obs, reward, done, truncated, _ = result
                done = done or truncated
            else:
                obs, reward, done, _ = result
            total_reward += reward
            time.sleep(delay)  # Pequena pausa para visualização
        print(f"Episódio {episode+1} finalizado com recompensa: {total_reward}")
    env.close()

# Carregar o modelo salvo
model = DQN()
model_path = "cartpole_model.pth"  # Caminho onde o modelo foi salvo
model.load_state_dict(torch.load(model_path))
model.eval()
print("Modelo carregado com sucesso!")

# Criar o ambiente com renderização
env = gym.make('CartPole-v1', render_mode='human')

# Ver o agente jogando
play_agent(env, model, num_episodes=5)
