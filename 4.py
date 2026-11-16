import pandas as pd

ds = pd.read_csv("./datasets/breastcancer.csv")

X = ds.iloc[:, :-1]
Y = ds.iloc[:, -1]


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

xtr, xte, ytr, yte = train_test_split(X, Y, test_size = 0.2)

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

mle = LogisticRegression()
mle.fit(xtr, ytr)
mle_predict = mle.predict(xte)


map_l1 = LogisticRegression(l1_ratio = 1, solver='liblinear') # L1
map_l1 = map_l1.fit(xtr, ytr)
map_l1_predict = map_l1.predict(xte)


map_l2 = LogisticRegression(l1_ratio = 0, solver='liblinear') # L2
map_l2 = map_l2.fit(xtr, ytr)
map_l2_predict = map_l2.predict(xte)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def metrics(name, ypredict, ytrue):
    print(f"-- {name} --")
    print(f"Acc : {accuracy_score(ytrue, ypredict)}")
    print(f"F1: {f1_score(ytrue, ypredict)}")
    print(f"Precision: {precision_score(ytrue, ypredict)}")
    print(f"Recall: {recall_score(ytrue, ypredict)}")

metrics("MLE", mle_predict, yte)
metrics("MAP_L1", map_l1_predict, yte)
metrics("MAP_L2", map_l2_predict, yte)
