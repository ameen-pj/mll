import pandas as pd

ds = pd.read_csv("./datasets/digits.csv", header=None)

X = ds.iloc[:, :-1]


from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cluster_sizes = [3, 5, 7, 9]

for cluster in cluster_sizes:
    model = KMeans(n_clusters = cluster)
    label = model.fit_predict(X)
    print(f" Cluster Size => {cluster}")
    print(f" Silhouette Score = {silhouette_score(X, label)}")
    print(f" Inertia = {model.inertia_}")

# print(ds)
