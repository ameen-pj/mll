import pandas as pd

ds = pd.read_csv("./datasets/wine_quality.csv")

# print(ds)
# print(ds.columns)

X = ds.iloc[:, :-1]
Y = ds.iloc[:, -1]

from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2, stratify = Y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

from sklearn.neural_network import MLPClassifier


layers = [(5,), (5, 10, ), (20, 10, 5,)]

accuracies = []

from sklearn.metrics import accuracy_score

for layer in layers:
    model = MLPClassifier(hidden_layer_sizes=layer, max_iter=10000)
    model.fit(xtr, ytr)
    ypr = model.predict(xte)
    print(f"{layer} : {accuracy_score(ytrue, ypr)}")
