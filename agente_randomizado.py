import gymnasium as gym
import time

# Cria o ambiente
env = gym.make('CartPole-v1', render_mode="human")

# Reseta o ambiente
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]  # Pegue apenas a observação

recompensa_total = 0
timesteps = 0

# Loop para ver o agente com ações aleatórias em ação
while True:
    # Ação aleatória
    acao = env.action_space.sample()
    
    # Executa a ação no ambiente
    obs, recompensa, terminou, truncado, info = env.step(acao)
    
    # Acumula a recompensa total
    recompensa_total += recompensa
    timesteps += 1
    
    # Renderiza o ambiente
    env.render()
    
    # Adiciona um pequeno atraso para visualizar melhor
    time.sleep(0.01)
    
    # Verifica se o episódio terminou e reseta o ambiente
    if terminou:
        print(f"Episódio terminado. Recompensa Total: {recompensa_total}, Timesteps: {timesteps}")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Pegue apenas a observação
        recompensa_total = 0
        timesteps = 0

# Fecha o ambiente
env.close()