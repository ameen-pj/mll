import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error


ds = pd.read_csv("./datasets/housing.csv")

X = ds[["AveRooms"]]
Y = ds["MedHouseVal"]

xtr, xte, ytr, yte = train_test_split(X, Y, test_size = 0.2, random_state=42)

sc = StandardScaler()

xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

model = LinearRegression()
model.fit(xtr, ytr)

ypr = model.predict(xte)

print("---- Normal Equation ----")
print(f"R2 Score : {r2_score(yte, ypr)}")
print(f"Mean Squared Error : {mean_squared_error(yte, ypr)}")

model = SGDRegressor()
model.fit(xtr, ytr)

ypr2 = model.predict(xte)

print("---- Gradient Descent----")
print(f"R2 Score : {r2_score(yte, ypr2)}")
print(f"Mean Squared Error : {mean_squared_error(yte, ypr2)}")

plt.scatter(xtr, ytr, color='r')
plt.plot(xte, ypr, label='Normal Equation')
plt.plot(xte, ypr2, label='Gradient Descent')
plt.xlabel("Average # of Rooms")
plt.ylabel("House Value")
plt.title("Linear Regression")
plt.legend()
plt.show()
