import numpy as np
import operator
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
from .base_model import Model

class SVM(Model):
    def __init__(self, data, max_epoch=1000, alpha = 0.0001):
        Model.__init__(self, data)
        self.max_epoch = max_epoch
        self.alpha = 0.0001
        self.weights = None
        
    def train(self):
        features = []
        self.weights = []
        
        lx, n_features = self.data["X_train"].shape
        for i in range(n_features):
            f = np.array(self.data["X_train"].iloc[:, i])
            features.append(np.reshape(f, (lx, 1)))
            self.weights.append(np.zeros((lx, 1)))
            
        y_train = list(map(lambda x: -1 if x == 0 else x, self.data["y_train"]))
        y_train = np.array(y_train)
        y_train = np.reshape(y_train, (len(y_train), 1))
        
        for epoch in range(1, self.max_epoch):
            y = self.weights[0] * features[0]
            for i in range(1, n_features):
                y += self.weights[i] * features[i]
            
            count = 0
            prod = y * y_train
            
            for val in prod:
                if (val >= 1):
                    cost = 0
                    for j in range(n_features):
                        w = self.weights[j]
                        self.weights[j] = w - self.alpha * (2 * 1/epoch * w)
                else :
                    cost = 1 - val
                    for j in range(n_features):
                        w = self.weights[j]
                        f = features[j]
                        self.weights[j] = w + self.alpha * (f[count] * y_train[count] - 2 * 1/epoch * w)
                count += 1
            
            if epoch % 100 == 0:
                print("epoch:", epoch)
                
    def predict_group(self, x):
        l, n_features = x.shape
        features = []
        index = list(range(l, self.data["X_train"].shape[0]))
        
        weights = list(map(
            lambda w: np.delete(w, index).reshape(l, 1),
            self.weights
        ))
        
        for i in range(n_features):
            f = np.array(x.iloc[:, i])
            features.append(f.reshape(l, 1))
        
        y = weights[0] * features[0]
        for i in range(1, n_features):
            y += weights[i] * features[i]
        return np.array(list(map(lambda val: int(val > 1), y)))
    
                    
                
        
