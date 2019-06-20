import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error


class PRegression:
    def __init__(self, degree):
        self.degree = degree
        self.p = preprocessing.PolynomialFeatures(degree)

    
    def fit(self, x, y):
        print(x, y)
        X = self.p.fit_transform(x, y)
        self.coeff = np.linalg.lstsq(X, y, rcond=None)[0]
        return self.coeff

    def predict(self, x):
        n = x.shape[0]
        X = self.p.fit_transform(x)
        Y = np.ndarray(n)
        for j in range(n):
            Y[j] = sum(X[j, i] * self.coeff[i]
                       for i in range(X.shape[1]))
        return Y


def popularity_plot(df, png="images/score_popularity_plot.png"):
    sns_plot = sns.scatterplot(
        x="popularity", y="score", data=df)
    sns_plot.get_figure().savefig(png)


def data_separator(X, y):
    train_x, test_x = [], []
    train_y, test_y = [], []
    i = 0
    for r, l in zip(X, y):
        if i % 10:
            train_x.append(r)
            train_y.append(l)
        else:
            test_x.append(r)
            test_y.append(l)
        i += 1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return (train_x, train_y), (test_x, test_y)

def main():
    metrics = ["score", "scored_by",
               "aired", "name", "popularity", "genre", "members", "rank"]

    df = pd.read_csv("datasets/anime.csv", sep=",")[metrics].dropna()

    popularity = np.array(df[['popularity']].to_numpy())

    score = df['score'].to_numpy()

    (train_data, train_value), (test_data, test_value) = \
        data_separator(popularity, score)

    plt.scatter(df['popularity'], df['score'], 5, 'g', 'o', alpha=0.8, label='data')

    PR = PRegression(1)
    PR.fit(train_data, train_value)
    res = PR.predict(test_data)

    plt.plot(test_data, res, color='red')
    e = mean_squared_error(res, test_value)

    print(e)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('images/polinomial_regression.png')


if __name__ == '__main__':
    main()