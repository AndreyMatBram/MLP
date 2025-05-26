import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def load_data():
    dictionary = {
        'Iris-setosa':-1,
        'Iris-versicolor':0, 
        'Iris-virginica':1
    }

    df = pd.read_csv('iris.data', header=None, sep=',')
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    df['class'] = df['class'].map(dictionary)

    df = shuffle(df, random_state=0)

    return df

def perceptron(MaxIt, a, X, d):
    w = np.zeros(X.shape[1])
    b = 0

    t = 1
    et = 1

    while t <= MaxIt and et > 0:
        et = 0
        for i in range(len(X)):

            y = np.sign(np.matmul(w, X[i]) + b) # y == 0 ? 0, ( y > 1 ? 1, -1)

            if d[i] != y:
                ei = d[i] - y
                w = w + a * ei * X[i]
                b = b + a * ei
                et = et + ei

    t = t + 1

    return w, b


def compare(WFinal, biasF, X, tag):

    result = [["", "", True]]

    for i in range(len(X)):
        y = np.sign(np.matmul(WFinal, X[i]) + biasF)

        if tag[i] == y:
            result.append([y, tag[i], True])
            print(result[i])
            continue
        result.append([y, tag[i], False])
        print(result[i])


def main():

    data = load_data()
    Wfinal, biasF = perceptron(MaxIt=100, a=0.1, X=data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values, d=data['class'].values)
    compare(Wfinal ,biasF ,X=data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values, tag=data['class'].values)

if __name__ == '__main__':
    main()