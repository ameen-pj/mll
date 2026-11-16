import pandas as pd

ds = pd.read_csv("./datasets/20newsgroups.csv").dropna()

X = ds['text']
Y = ds['target']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2)


cv = CountVectorizer(stop_words='english')
xtr = cv.fit_transform(xtr)
xte = cv.transform(xte)

from sklearn.naive_bayes import MultinomialNB

alphas = [0, 0.1, 1, 2]

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

for alpha in alphas:
    model = MultinomialNB(alpha=alpha)
    model.fit(xtr, ytr)
    ypredict = model.predict(xte)
    print(f"---- alpha = {alpha} ----")
    print(f"Acc : {accuracy_score(ytrue, ypredict)}")
    print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
    print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
    print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")
