import pandas as pd

ds = pd.read_csv("./datasets/diabetes.csv")

X = ds[["bmi"]]
Y = ds["target"]

from sklearn.model_selection import train_test_split

xtr, xte , ytr, yte = train_test_split(X, Y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)


from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

lr = LinearRegression()
lr.fit(xtr, ytr)
lry = lr.predict(xte)

lasso = LassoCV(cv=5)
lasso.fit(xtr, ytr)
lassoy = lasso.predict(xte)


ridge= RidgeCV(cv=5)
ridge.fit(xtr, ytr)
ridgey = ridge.predict(xte)


from sklearn.metrics import mean_squared_error, r2_score

print(f"r2 : {r2_score(yte, lry)}")
print(f"mse: {mean_squared_error(yte, lry)}")


print(f"r2 : {r2_score(yte, lassoy)}")
print(f"mse: {mean_squared_error(yte, lassoy)}")


print(f"r2 : {r2_score(yte, ridgey)}")
print(f"mse: {mean_squared_error(yte, ridgey)}")

