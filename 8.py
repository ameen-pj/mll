from time import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

ds = pd.read_csv('./dataset/mnist.csv')

X = ds.iloc[:, 1: -1]
Y = ds.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=.2)

kvals = [3,5,7]
accuracies = []
training_times = []
f1 = []
recall = []


for k in kvals:
    knn = KNeighborsClassifier(n_neighbors=k)
    start = time()
    knn.fit(x_train, y_train)
    stop = time()
    y_pred = knn.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    training_times.append(stop - start)
    f1.append(f1_score(y_test, y_pred, average='weighted'))
    recall.append(recall_score(y_test, y_pred, average='weighted'))

# Print the accuracies for each k value
for k, accuracy in zip(kvals, accuracies):
    
    print(f'Accuracy for k={k}: {accuracy:.4f} Training time: {training_times[kvals.index(k)]:.4f} seconds')
    print(f'F1 Score for k={k}: {f1[kvals.index(k)]:.4f}')
    print(f'Recall for k={k}: {recall[kvals.index(k)]:.4f}')