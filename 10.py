import pandas as pd
from pandas.api.types import is_string_dtype


ds = pd.read_csv("./datasets/adult_income.csv")
ds = ds.dropna()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ds.columns:
    if is_string_dtype(ds[col]):
        ds[col] = le.fit_transform(ds[col])

X = ds.iloc[:, : -1]
Y = ds.iloc[:, -1]


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model = LogisticRegression()
model.fit(xtr , ytr)
ypredict = model.predict(xte)

print("=LR=")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")

model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(xtr , ytr)
ypredict = model.predict(xte)

print("=id3=")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")

