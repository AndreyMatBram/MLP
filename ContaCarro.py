import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Criar um DataFrame de exemplo (substitua pelo seu próprio DataFrame)

df = pd.read_csv('car.data', header=None, sep=',')
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


# 1. Contar os valores de cada classe
contagem = df['class'].value_counts().reset_index()


# 3. Plotar a matriz de contagem
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Criar um heatmap com os dados de contagem
heatmap_data = contagem.set_index('class').T  # Transpor para formato de matriz

sns.heatmap(heatmap_data, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            cbar=False,
            linewidths=.5)

plt.title("Contagem de Valores por Classe")
plt.xticks(rotation=45)
plt.yticks([])  # Remover o rótulo do eixo Y pois não é necessário

plt.tight_layout()
plt.show()