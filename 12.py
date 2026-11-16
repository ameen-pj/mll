import pandas as pd

ds = pd.read_csv("./datasets/fashion_mnist.csv")

# print(ds)

X = ds.iloc[:, 1:]
Y = ds.iloc[:, 0]

# print(Y)
# print(ds)
# print(ds.columns)

from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

from sklearn.preprocessing import StandardScaler


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model = SVC(kernel='linear')
model.fit(xtr, ytr)
ypredict = model.predict(xte)


print("= Linear =")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")


model = SVC(kernel='rbf')
model.fit(xtr, ytr)
ypredict = model.predict(xte)


print("= rbf =")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")


model = SVC(kernel='poly')
model.fit(xtr, ytr)
ypredict = model.predict(xte)


print("= poly =")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")

