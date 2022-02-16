from model.knn import KNN
from model.naive_bayes import NaiveBayes
from model.svm import SVM
import pandas as pd
import math

# pre-process data


def splitData(df, train_frac=0.8):
    nrow = df.shape[0]
    train_len = math.floor(nrow * train_frac)
    test_len = nrow - train_len
    return df.head(train_len), df.tail(test_len)


raw_df = pd.read_csv("mlb.txt", sep=" ")

train, test = splitData(raw_df)
X_train = train[['Height', 'Age', 'Weight']]
y_train = train['Position'].map(lambda x: int(x == 'Catcher'))
X_test = test[['Height', 'Age', 'Weight']]
y_test = test['Position'].map(lambda x: int(x == 'Catcher'))

data = {
    "X_train": X_train.to_numpy(),
    "y_train": y_train.to_numpy(),
    "X_test": X_test.to_numpy(),
    "y_test": y_test
}

# test the model
model = KNN(data)
print("KNN test accuracy ", model.test())
print("========================")

model = NaiveBayes(data)
print("Naive Bayes test accuracy ", model.test())
print("========================")

data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test
}

model = SVM(data)
model.train()
print("SVM test accuracy ", model.test())
