from classifier import classifier
import numpy as np
from collections import Counter

class KNN(classifier):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):
        pred = []

        for i in range(len(X)):

            distances = []
            targets = []

            x_test = X.iloc[i, :]

            for m in range(len(self.X)):
                distance = np.sqrt(np.sum(np.square(x_test - self.X.iloc[m, :])))
                distances.append([distance, m])

            distances = sorted(distances)

            for j in range(self.k):
                index = distances[j][1]
                targets.append(self.Y[index])

            target = Counter(targets).most_common(1)[0][0]
            pred.append(target)

        return pred
