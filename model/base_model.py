from sklearn.metrics import accuracy_score
import pandas as pd

class Model:
    def __init__(self, data):
        self.data = data
    
    def train(self):
        raise NotImplementedError()
        
    def predict(self, x):
        raise NotImplementedError()
    
    def predict_group(self, X):
        y_pred = [self.predict(x) for x in X]
        return y_pred

    def test(self):
        predictions = self.predict_group(self.data["X_test"])
        return accuracy_score(self.data["y_test"], predictions)
    
    def compare_df(self):
        predictions = self.predict_group(self.data["X_test"])
        return pd.DataFrame.from_dict({
            "real": self.data["y_test"],
            "predict": predictions
        })
