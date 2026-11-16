import pandas as pd

ds = pd.read_csv("./datasets/mnist.csv")

#print(ds)
#print(ds.columns)

X = ds.iloc[:, 1:] / 255.0
Y = ds.iloc[:, 0]

from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2, stratify = Y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

from sklearn.neural_network import MLPClassifier


act = ['logistic', 'relu', 'tanh']


from sklearn.metrics import accuracy_score

for a in act:
    model = MLPClassifier(hidden_layer_sizes=(5, 3, ), activation=a)
    model.fit(xtr, ytr)
    ypr = model.predict(xte)
    print(f"{a} : {accuracy_score(ytrue, ypr)}")
