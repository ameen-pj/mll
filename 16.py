import pandas as pd

ds = pd.read_csv("./datasets/Mall_Customers.csv")

print(ds.columns)
print(ds)

X = ds.iloc[:, :]

from sklearn.preprocessing import StandardScaler, LabelEncoder

X["Genre"] = LabelEncoder().fit_transform(X["Genre"])

X = StandardScaler().fit_transform(X)

from sklearn.cluster import KMeans, AgglomerativeClustering

model1 = KMeans()
label1 = model1.fit_predict(X)


model2 = AgglomerativeClustering()
label2 = model2.fit_predict(X)

from sklearn.metrics import silhouette_score

print("-- KMeans --")
print(f"Inertia : {model1.inertia_}")
print(f"SC: {silhouette_score(X, label1)}")


print("-- Agglomerative Clustering --")
print(f"SC: {silhouette_score(X, label2)}")

import matplotlib.pyplot as plt

plt.subplot(1,2, 1)
plt.scatter(X[:,3], X[:,4], c = label1)
plt.subplot(1,2, 2)
plt.scatter(X[:,3], X[:, 4], c = label2)
plt.tight_layout()
plt.show()
