import pandas as pd

ds = pd.read_csv("./datasets/20newsgroups.csv")
ds = ds.dropna()

# print(ds)
# print(ds.columns)

X = ds['text']
Y = ds['category']

from sklearn.model_selection import train_test_split

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')


xtr = cv.fit_transform(xtr)
xte = cv.transform(xte)


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model = MultinomialNB()
model.fit(xtr, ytr)
ypredict = model.predict(xte)

print("-- MultinomialNB --")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")


model = BernoulliNB()
model.fit(xtr, ytr)
ypredict = model.predict(xte)

print("-- MultinomialNB --")
print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")
