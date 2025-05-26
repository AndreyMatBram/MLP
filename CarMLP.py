import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def load_data():

# buying:   vhigh, high, med, low.
# maint:    vhigh, high, med, low.
# doors:    2, 3, 4, 5more.
# persons:  2, 4, more.
# lug_boot: small, med, big.
# safety:   low, med, high.
# class:    unacc, acc, good, vgood.

    dic_buying = {
        'vhigh': 0,
        'high': 1,
        'med': 2,
        'low': 3
    }
    dic_maint = {
        'vhigh': 0,
        'high': 1,
        'med': 2,
        'low': 3
    }
    dic_doors = {
        '2': 0,
        '3': 1,
        '4': 2,
        '5more': 3
    }
    dic_persons = {
        '2': 0,
        '4': 1,
        'more': 2
    }
    dic_lug_boot = {
        'small': 0,
        'med': 1,
        'big': 2
    }
    dic_safety = {
        'low': 0,
        'med': 1,
        'high': 2
    }
    dic_class = {
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3
    }

    df = pd.read_csv('car.data', header=None, sep=',')
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    df['buying'] = df['buying'].map(dic_buying)
    df['maint'] = df['maint'].map(dic_maint)
    df['doors'] = df['doors'].map(dic_doors)
    df['persons'] = df['persons'].map(dic_persons)
    df['lug_boot'] = df['lug_boot'].map(dic_lug_boot)
    df['safety'] = df['safety'].map(dic_safety)
    df['class'] = df['class'].map(dic_class)

    df = shuffle(df, random_state=0)

    X = df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']].values

    d = df['class'].values

    return X, d

# Dividir os dados em treino, validação e teste
def split_data(data, output, train_size=0.2, validation_size=0.2, test_size=0.2):
    
    assert train_size + validation_size + test_size == 1, "Os tamanhos de treino, validação e teste devem somar 1"
    
    train_size = int(len(data) * train_size)
    validation_size = int(len(data) * validation_size)

    data_train = data[:train_size]
    output_train = output[:train_size]

    data_validation = data[train_size:train_size + validation_size]
    output_validation = output[train_size:train_size + validation_size]

    data_test = data[train_size + validation_size:]
    output_test = output[train_size + validation_size:]

    return data_train, output_train, data_validation, output_validation, data_test, output_test



def plot_all_matriz_confusao(conf_matrices, accuracies, split_names, title='Matriz de Confusão'):
    plt.figure(figsize=(18, 5))  # Figura mais larga para 3 subplots
    
    for i, (conf_matrix, accuracy, split_name) in enumerate(zip(conf_matrices, accuracies, split_names)):
        plt.subplot(1, 3, i+1)  # 1 linha, 3 colunas, posição i+1
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['unacc', 'acc', 'good', 'vgood'],
                    yticklabels=['unacc', 'acc', 'good', 'vgood'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{split_name} : {accuracy * 100:.2f}%')
    
    plt.tight_layout()  # Ajusta o espaçamento entre subplots
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)  # Ajusta o espaço para o título
    plt.show()

def plot_loss(history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.show()


def normalize_with_reference(X_ref, X_to_normalize):
    X_min = X_ref.min(axis=0)
    X_max = X_ref.max(axis=0)
    return (X_to_normalize - X_min) / (X_max - X_min)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, random_seed=None):

        if random_seed is not None:
            np.random.seed(random_seed)

        #atribuindo os atributos de entrado como atributos da classe
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #inicializando os pesos e bias
        #os pesos são inicializados aleatoriamente entre 0 e 1
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

        self.bias_hidden = np.random.rand(self.hidden_size)
        self.bias_output = np.random.rand(self.output_size)

    def forward(self, X):

        # Realizar a propagação para frente
        # Calcular a saída da camada oculta
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.final_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output
    
    def backward(self, X, y, output):	

        # Calcular o erro e a derivada do erro
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        # Atualizar os pesos e bias
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate

        self.weights_input_hidden += np.dot(X.T, hidden_layer_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0) * self.learning_rate
    
    def train(self, X, y, epochs=1000):
        self.history = []
        # Realizar o treinamento da rede neural pela quantidade de épocas especificada
        for _ in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)  # Erro quadrático médio
            self.history.append(loss)
            self.backward(X, y, output)
        return self.history

    def predict(self, X):
        # Realizar a previsão com a rede neural
        output = self.forward(X)
        return np.argmax(output, axis=1)



def main():

    data, results = load_data()

    # Convertendo os resultados para uma matriz de saída
    output = np.zeros((len(results), 4))
    for i in range(len(results)):
        output[i][results[i]] = 1 # 0, 1, 2 => [1, 0, 0], [0, 1, 0], [0, 0, 1]

    # Dividindo os dados em treino, validação e teste

    train_size = 0.6        #os sizes de treino, validação e teste devem somar 1
    validation_size = 0.2 
    test_size = 0.2

    randSeed = 36  # Semente aleatória para reprodutibilidade
    lr = 0.017 # Taxa de aprendizado (consegui esse valor do grafico de LR x Acurácia)

    data_train, output_train, data_validation, output_validation, data_test, output_test = split_data(data, output, train_size, validation_size, test_size)

    split_names = ["Treino", "Validação", "Teste"]
    
    # === NORMALIZAÇÃO
    data_train_norm = normalize_with_reference(data_train, data_train)
    data_validation_norm = normalize_with_reference(data_train, data_validation)
    data_test_norm = normalize_with_reference(data_train, data_test)

    # === MLP COM NORMALIZAÇÃO
    
    mlp_norm = MLP(input_size=6, hidden_size=4, output_size=4, learning_rate=lr, random_seed=randSeed)

    history_norm = mlp_norm.train(data_train_norm, output_train, epochs=1000)
    plot_loss(history_norm, title='Gráfico de perda COM normalização')  # Gráfico de perda COM normalização

    conf_matrices_norm = []
    accuracies_norm = []
    datasets_norm = [(data_train_norm, output_train), 
                     (data_validation_norm, output_validation), 
                     (data_test_norm, output_test)]

    for (data_split, output_split) in datasets_norm:
        output_true = np.argmax(output_split, axis=1)
        output_pred = mlp_norm.predict(data_split)
        accuracy = np.mean(output_true == output_pred)
        accuracies_norm.append(accuracy)
        cm = confusion_matrix(output_true, output_pred)
        conf_matrices_norm.append(cm)

    plot_all_matriz_confusao(conf_matrices_norm, accuracies_norm, split_names, title='Matriz de Confusão COM normalização')  # Gráfico de matriz de confusão COM normalização

if __name__ == '__main__':
    main()