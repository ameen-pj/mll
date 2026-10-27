import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

ds = pd.read_csv('./dataset/digits.csv')

X = ds.iloc[:, :-1]
Y = ds['target']


clusters = [3, 5, 7, 9]

sc = []
inertia = []


for i,cluster in enumerate(clusters):
    km = KMeans(n_clusters = cluster, max_iter=300)
    labels = km.fit_predict(X, Y)
    sc.append(silhouette_score(X, labels))
    inertia.append(km.inertia_)


for s, i in zip( sc, inertia):
    print(f"SC : {s} | Inertia : {i}")
