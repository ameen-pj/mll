import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error


ds = pd.read_csv("./datasets/boston_housing.csv")

X = ds.iloc[:, [5]] # Avg. No or Rooms
Y = ds.iloc[:, [-1]]

xtr, xte, ytr, yte = train_test_split(X, Y, test_size = 0.2)

degrees = [1, 2, 3, 4, 5]
tr_errors = []
te_errors = []

for degree in degrees:
    poly = PolynomialFeatures(degree)
    xtr_poly = poly.fit_transform(xtr)
    xte_poly = poly.transform(xte)

    sc = StandardScaler()

    xtr_poly = sc.fit_transform(xtr_poly)
    xte_poly = sc.transform(xte_poly)

    lg = LinearRegression()
    lg.fit(xtr_poly, ytr)

    tr_pred = lg.predict(xtr_poly)
    te_pred = lg.predict(xte_poly)

    tr_err = mean_squared_error(ytr, tr_pred)
    te_err = mean_squared_error(yte, te_pred)

    tr_errors.append(tr_err)
    te_errors.append(te_err)


plt.title("Bias-Variance Tradeoff")
plt.plot(degrees, tr_errors, label='Training Error')
plt.plot(degrees, te_errors, label='Validation Error')
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.savefig("plots/20.png")
plt.show()