import pandas as pd

ds = pd.read_csv("./datasets/titanic.csv")

ds = ds[["survived", "pclass", "sex",  "age",  "cabin"]].dropna()

X = ds[["pclass", "sex", "age", "cabin"]]
Y = ds[["survived"]]


# Encoding String labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in X.columns:
    X[col] = le.fit_transform(X[col])

# Training & Test Split
from sklearn.model_selection import train_test_split

xtr, xte, ytr, yte = train_test_split(X, Y, test_size=0.2)

# Bagging - Bootstrap Aggregating
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bc = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, bootstrap=True)
bc.fit(xtr, ytr)

bc_predict = bc.predict(xte)


# Boosting
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
abc.fit(xtr, ytr)
abc_predict = abc.predict(xte)

# Classification metircs
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def printMetrics(name, ytrue, ypredict):
    print(f"-----{name}-----")
    print(f"Acc : {accuracy_score(ytrue, ypredict)}")
    print(f"f1 : {f1_score(ytrue, ypredict, average='weighted')}")
    print(f"precision : {precision_score(ytrue, ypredict, average='weighted')}")
    print(f"Recall : {recall_score(ytrue, ypredict, average='weighted')}")




printMetrics("Bagging", yte,bc_predict)
printMetrics("Boosting", yte,abc_predict)