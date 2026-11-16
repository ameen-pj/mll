import pandas as pd

ds = pd.read_csv("./datasets/pimadiabetes.csv")


X = ds.iloc[:, :-1]
Y = ds.iloc[:, -1]


from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue =  train_test_split(X, Y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

lr = LogisticRegression(max_iter=100000)
lr.fit(xtr, ytr)
ypredict = lr.predict(xte)

print("-- LR without Feature Scaling --")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

lr = LogisticRegression(max_iter=100000)
lr.fit(xtr, ytr)
ypredict = lr.predict(xte)


print("-- LR with Feature Scaling --")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")

