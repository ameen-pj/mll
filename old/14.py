import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

ds = pd.read_csv('./dataset/fashion_mnist.csv', low_memory=False)


X = ds.iloc[:100, 1:].astype(float) / 255.0
Y = ds.iloc[:100, 0]

xtr, xte, ytr, yte = train_test_split(X, Y, random_state=32, test_size=0.2)

activation_functions = ['identity', 'logistic', 'tanh', 'relu']
for activation in activation_functions:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation=activation, max_iter=5, random_state=42)
    mlp.fit(xtr, ytr)
    accuracy = mlp.score(xte, yte)
    print(f'Activation function: {activation}, Accuracy: {accuracy:.4f}')





