import pandas as pd

ds = pd.read_csv("./datasets/iris.csv")

X = ds.iloc[:, [0, 1]]
Y = ds.iloc[:, -1]
Y = (Y == 0).astype(int)

# print(Y)

# print(ds)
# print(ds.columns)

from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

from sklearn.preprocessing import StandardScaler


from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(xtr, ytr)
ypredict = model.predict(xte)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")

import matplotlib.pyplot as plt

print(model.coef_)
print(model.intercept_)

b = model.intercept_[0]
w1 = model.coef_[0][0]
w2 = model.coef_[0][1]

# w1x1 + w2x2  + b = 0

import numpy as np


x = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 1000)
y1 = (1 - (b + w1*x)) / w2
y2 = (-1 - (b + w1*x)) / w2
y3 = (-b -w1*x) / w2

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = Y)
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()

