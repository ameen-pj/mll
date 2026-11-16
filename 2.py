import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ds = pd.read_csv("./datasets/autompg.csv")

X = ds[["displacement"]]
Y = ds["class"]

xtr, xte, ytr, yte = train_test_split(X, Y, test_size = 0.2)

print("--- Linear Regression ---")

sc = StandardScaler()
xtr_sc = sc.fit_transform(xtr)
xte_sc = sc.transform(xte)

model = LinearRegression()
model.fit(xtr_sc, ytr)

ypr_lr = model.predict(xte_sc)

print(f"r2 : {r2_score(yte, ypr_lr)}")
print(f"mse: {mean_squared_error(yte, ypr_lr)}")


print("---- Polynomial Regression ----")

degrees = [2, 3, 5, 7, 9]
yprs = []
mses = []
r2s = []

for d in degrees:
    p = PolynomialFeatures()
    xtr_p = p.fit_transform(xtr_sc)
    xte_p = p.transform(xte_sc)

    model = LinearRegression()
    model.fit(xtr_p, ytr)

    ypr = model.predict(xte_p)
    mses.append(mean_squared_error(yte, ypr))
    r2s.append(r2_score(yte, ypr))
    yprs.append(ypr)


print(f"r2 : {np.mean(r2s)}")
print(f"mse: {np.mean(mses)}")


plt.scatter(xtr_sc, ytr)
for i in range(len(degrees)):
    plt.plot(xte_sc, yprs[i], label=str(degrees[i]))

plt.plot(xte_sc, ypr_lr, label='Linear Regression')
plt.xlabel("Engine Displacement")
plt.ylabel("MPG")
plt.legend()
plt.show()
