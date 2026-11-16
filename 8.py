import pandas as pd

ds = pd.read_csv("./datasets/fashion_mnist.csv")
ds = ds.dropna()

# print(ds)
# print(ds.columns)

X = ds.iloc[:, 1:] / 255.0
Y = ds.iloc[:, 0]

from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model = KNeighborsClassifier()
model.fit(xtr, ytr)
ypredict = model.predict(xte)

print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")
