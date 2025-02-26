import gymnasium as gym
import numpy as np
import torch
import ale_py
from stable_baselines3 import DQN
import time

# Importando as classes necessárias do seu código original
from treinmanento_mlp import RamPreprocessing, MCTSMLPPong  # Assuma que seu código original está em mcts_mlp_pong.py

def play_pong():
    # Configura o ambiente
    gym.register_envs(ale_py)
    env = gym.make("Pong-ram-v4", render_mode="human")  # Adiciona render_mode="human" para visualização
    env = RamPreprocessing(env)
    
    try:
        # Cria o agente MCTS-MLP e carrega o modelo treinado
        agent = MCTSMLPPong(env=env, model_path="pong_mlp_dqn.zip")
        
        while True:  # Loop para permitir múltiplos jogos
            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            print("Iniciando novo jogo...")
            
            while not done:
                # Usa o MCTS para escolher a ação
                action = agent.choose_action(state, num_simulations=25)  # Reduzido para maior velocidade
                
                # Executa a ação
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                step += 1
                
                # Pequena pausa para melhor visualização
                time.sleep(0.02)
                
                if truncated:
                    break
           
            # Pergunta se quer jogar novamente
            play_again = input("\nDeseja jogar novamente? (s/n): ").lower()
            if play_again != 's':
                break
                
    except KeyboardInterrupt:
        print("\nJogo interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro durante a execução: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    play_pong()