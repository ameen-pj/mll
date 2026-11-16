import pandas as pd

ds = pd.read_csv("./datasets/OnlineRetail.csv", encoding='latin1')
ds = ds.dropna()
ds = ds[["InvoiceNo", 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']]
ds['Amt'] = ds['Quantity'] * ds['UnitPrice']


ds = ds.groupby(by='CustomerID').agg(
        invoice_count=('InvoiceNo', 'nunique'),
        stock_count=('StockCode', 'nunique'),
        total_qty=('Quantity', 'sum'),
        avg_unit_price=('UnitPrice', 'mean'),
        amt_total=('Amt', 'sum')
)

threshold = ds['amt_total'].quantile(0.80)

ds["target"] = (ds['amt_total'] >= threshold).astype(int)
ds = ds.drop(columns=['amt_total'])


from sklearn.model_selection import train_test_split

X = ds.iloc[:, :-1]
Y = ds.iloc[:, -1]

xtr, xte, ytr, ytrue = train_test_split(X, Y, test_size = 0.2)


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


model = DecisionTreeClassifier(max_depth=3)
model.fit(xtr , ytr)
ypredict = model.predict(xte)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print(f"Acc : {accuracy_score(ytrue, ypredict)}")
print(f"F1: {f1_score(ytrue, ypredict, average='weighted')}")
print(f"Precision: {precision_score(ytrue, ypredict, average='weighted')}")
print(f"Recall: {recall_score(ytrue, ypredict, average='weighted')}")
plot_tree(model, feature_names=X.columns, class_names=['Not High Value', 'High Value'])


plt.show()
