import numpy as np
import pandas as pd

# Carrega as observações
observacoes = np.load("observacoes.npy")

# Cria um DataFrame com as observações
# Como Pong-ram usa 128 bytes de RAM como observação, vamos nomear cada coluna
colunas = [f'RAM_{i}' for i in range(observacoes.shape[1])]
df = pd.DataFrame(observacoes, columns=colunas)

# Adiciona uma coluna de índice temporal
df.insert(0, 'Timestep', range(len(df)))

# Mostra as primeiras linhas da tabela
print("\nPrimeiras 5 observações:")
print(df.head())

 # -1 por causa da coluna Timestep