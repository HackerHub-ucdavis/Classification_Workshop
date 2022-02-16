import numpy as np
import operator
import pandas as pd
from collections import Counter
from .base_model import Model


class KNN(Model):
    def __init__(self, data, k=3):
        Model.__init__(self, data)
        self.k = 3

    def train(self):
        print("No need to train")

    def predict(self, x):
        # compute the distances
        distances = [self._distance(x, x_train) for x_train in self.data["X_train"]]
        
        # get the k-nearest samples, labels
        # sort distances, return an array of indices, get only first k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.data["y_train"][i] for i in k_indices]
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]
    
    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
