from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

ds = fetch_california_housing()

Y = ds.target
X = ds.data[:, [0]]

xtr, xte, ytr, yte = train_test_split(X, Y, random_state=42, test_size=0.2)

sc = StandardScaler()

xtr_scaled = sc.fit_transform(xtr)
xte_scaled = sc.transform(xte)

lr = LinearRegression()
lr.fit(xtr_scaled, ytr)
lr_pr = lr.predict(xtr_scaled)


gd = SGDRegressor(max_iter=1000)
gd.fit(xtr_scaled, ytr)
gd_pr = gd.predict(xtr_scaled)


print("-- Linear Regression --")
print("Training MSE:", mean_squared_error(ytr, lr_pr))
print("Training R2:", r2_score(ytr, lr_pr))

print("\n-- Gradient Descent Regression --")
print("Training MSE:", mean_squared_error(ytr, gd_pr))
print("Training R2:", r2_score(ytr, gd_pr))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(xtr_scaled, ytr, color='blue', label='Training Data')
plt.plot(xtr_scaled, lr_pr, color='red', label='Regression Line')
plt.title('Linear Regression on Training Data')
plt.xlabel('Scaled Feature')
plt.ylabel('Target')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(xtr_scaled, ytr, color='blue', s=2, label='Training Data')
plt.plot(xtr_scaled, gd_pr, color='green', ls='--', label='Regression Line')
plt.title('Gradient Descent Regression on Training Data')
plt.xlabel('Scaled Feature')
plt.ylabel('Target')
plt.legend()

plt.show()