import pandas as pd

ds = pd.read_csv("./datasets/fashion_mnist.csv")

# print(ds)
# print(ds.columns)

X = ds.iloc[:, 1:] / 255.0
Y = ds.iloc[:, 0]

from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2, stratify = Y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

from sklearn.neural_network import MLPClassifier


lrs = [0.1, 0.01, 0.02]
bss = [100, 200, 500, 600]



from sklearn.metrics import accuracy_score

for lr in lrs:
    for bs in bss:
        model = MLPClassifier(
                hidden_layer_sizes=(5,), 
                learning_rate = 'constant',
                learning_rate_init = lr,
                batch_size = bs
        )
        model.fit(xtr, ytr)
        ypr = model.predict(xte)
        print(f"{lr} | {bs} : {accuracy_score(ytrue, ypr)}")
