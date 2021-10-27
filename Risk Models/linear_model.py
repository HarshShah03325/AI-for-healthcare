from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils1 import lr_model, cindex, add_interactions, make_standard_normal
import pandas as pd

model = LinearRegression()

X = pd.read_csv('X_data.csv',index_col=0)
y_df = pd.read_csv('y_data.csv',index_col=0)
y = y_df['y']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)
X_train, X_test = make_standard_normal(X_train_raw, X_test_raw)

model_X = lr_model(X_train, y_train)


scores = model_X.predict_proba(X_test)[:, 1]
c_index_X_test = cindex(y_test.values, scores)


X_train_int = add_interactions(X_train)
X_test_int = add_interactions(X_test)


model_X_int = lr_model(X_train_int, y_train)


scores_X = model_X.predict_proba(X_test)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X)

scores_X_int = model_X_int.predict_proba(X_test_int)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X_int)

print(f"c-index on test set without interactions is {c_index_X_test:.4f}")
print(f"c-index on test set with interactions is {c_index_X_int_test:.4f}")