from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

dataset = pd.read_csv('./dataset/diabetes.csv')

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

lr = LogisticRegression(max_iter=1000)

print("---- Without Feature Scaling -----")

lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)

print(classification_report(y_test, y_predict))


print("---- Feature Scaling -----")

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


lr = LogisticRegression(max_iter=1000)

lr.fit(x_train_scaled, y_train)
y_predict = lr.predict(x_test_scaled)

print(classification_report(y_test, y_predict))
