import pandas as pd

ds = pd.read_csv("./datasets/iris.csv")

X = ds.iloc[:, :-1]
Y = ds.iloc[:, -1]

# K-Fold Cross Validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold

model = DecisionTreeClassifier()

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy = cross_val_score(model, X, Y, scoring='accuracy', cv=kfold)
f1 = cross_val_score(model, X, Y, scoring='f1_weighted', cv=kfold)

import numpy as np

print(f"acc : {np.mean(accuracy)} | f1 : {np.mean(f1)}")


# Resampling

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score

accs = []
f1s = []

for i in range(50,101):
    x_sample, y_sample = resample(X, Y, random_state = i, replace=True)
    model = DecisionTreeClassifier()
    model.fit(x_sample, y_sample)
    l = model.predict(X)
    accs.append(accuracy_score(Y, l))
    f1s.append(f1_score(Y, l, average='weighted'))

print(f"Acc : {np.mean(accs)}")
print(f"f1 : {np.mean(f1s)}")
