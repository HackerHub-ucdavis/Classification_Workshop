import numpy as np
import operator
import pandas as pd
from collections import Counter
from .base_model import Model


class NaiveBayes(Model):
    def __init__(self, data):
        Model.__init__(self, data)
        
        X = self.data["X_train"]
        y = self.data["y_train"]

        n_samples, n_features = X.shape
        # find unique elements of y, use as classes
        self.classes = np.unique(y)

        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:    # for each class
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)    # find mean of each column
            self.var[c] = X_c.var(axis=0)
            # frequency of this class in training sample
            self.priors[c] = X_c.shape[0] / float(n_samples)

    def train(self):
        print("No need to train")

    def predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = self.priors[c]
            class_conditional = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_name, x):
        '''probability density function'''
        mean = self.mean[class_name]
        var = self.var[class_name]

        numerator = np.exp(- (x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator
