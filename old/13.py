import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Recommended for Neural Networks

ds = pd.read_csv('./dataset/wine_quality.csv')

X = ds.iloc[:, :-1]
Y = ds.iloc[:, -1]

xtr, xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=32)

# 1. Scale features so all inputs are on a similar scale (mean=0, variance=1)
scaler = StandardScaler()
xtr_scaled = scaler.fit_transform(xtr)
xte_scaled = scaler.transform(xte)

archs = [(5,), (10,), (5, 10,)]
accs = []

for arch in archs:
    # 2. Added max_iter and random_state for reproducible, stable convergence
    model = MLPClassifier(hidden_layer_sizes=arch, activation='tanh', max_iter=500, random_state=32)
    model.fit(xtr_scaled, ytr)  # Train on scaled training data
    ypr = model.predict(xte_scaled)  # Predict on scaled test data

    accs.append(accuracy_score(yte, ypr))

# 3. Fixed zip order: arch comes first, then acc
for arch, acc in zip(archs, accs):
    print(f"Accuracy for architecture {arch}: {acc:.4f}")