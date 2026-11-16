from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
ds = load_iris()

X = ds.data[:, :2]  # Use only first two features
Y = (ds.target == 0).astype(int)

# Split dataset
xtr, xte, ytr, yte = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Train SVM
classifier = SVC(kernel='linear', max_iter=1000)
classifier.fit(xtr, ytr)

# Predict and calculate accuracy
ypr = classifier.predict(xte)
print("Accuracy:", accuracy_score(yte, ypr))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')

# Get decision boundary
w1 = classifier.coef_[0][0]
w2 = classifier.coef_[0][1]
b = classifier.intercept_[0]

# w1x1 + w2x2 + b = 0

x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

db = (-b -w1*x) / w2
mp = (1 - w1*x - b) / w2
mn = (-1 - w1*x - b) / w2
plt.plot(x, db, ls = '-', label='Decision Boundary')
plt.plot(x, mp, ls = '--', label = '+ve Margin Boundary')
plt.plot(x, mn, ls = '--', label = '-ve Margin Boundary')


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM Decision Boundary")
plt.legend()
plt.show()
