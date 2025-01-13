import gymnasium as gym
import time
import matplotlib.pyplot as plt

# Caso alguém queira ler sobre meu get started.

lista_visualizacoes  =  [] # irá salvar todas as visualizações renderizadas

env = gym.make('CartPole-v1',render_mode="human")
obs = env.reset() 
# Até aqui, temos somente o estado inicial do ambiente definido.

for i in range(100):
    env.render() # renderiza o ambiente atual em cada iteração abrindo uma tela de visualização do ambiente
    lista_visualizacoes.append(env.render()) # salva cada renderização para analisarmos depois. O uso de
    
    # mode='rgb_array' retorna um array numpy com os valores RGB de cada posição, e o imshow do matplotlib os exibe na tela.
    time.sleep(0.2) # dá um atraso de 0.2s para mostrar mais lentamente
    acao = env.action_space.sample() # gera uma nova ação aleatória
    print(obs) # imprime o estado atual do sistema
    obs, recompensa, terminou,truncado, info = env.step(acao) # a partir de cada ação, vai para um novo estado, recebe recompensa, 
    # diz se o episódio terminou (done) e traz um dicionário com outras informações se houver 
    if terminou: # se terminou um episódio (nesse caso, se a barra caiu ou se o carrinho foi pra fora dos limites da figura)
        print("Episódio finalizado depois de {} timesteps".format(i+1))
        break

env.close() # fecha a janela