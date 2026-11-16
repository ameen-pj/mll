import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

ds = pd.read_csv('./dataset/Mall_Customers.csv')

X = ds.iloc[:, 3:]
Y = ds.iloc[:, -1]

kmeans = KMeans()
labels = kmeans.fit_predict(X)

print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X, labels))


acluster = AgglomerativeClustering(n_clusters=5)
a_labels = acluster.fit_predict(X)
print("Agglomerative Clustering Silhouette Score:", silhouette_score(X, a_labels))

plt.subplot(1, 2, 1)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")


plt.subplot(1, 2, 2)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.show()