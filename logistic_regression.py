import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = pd.read_csv("data/punctuation_data.csv")

data["generation"] = 0

for i in range(len(data)):
    if data["birth_year"][i] in range(1980, 1997):
        data["generation"][i] = "M"
    if data["birth_year"][i] in range(1997, 2013):
        data["generation"][i] = "Z"

final_data = data[data["generation"] != 0]
print(final_data.head())

X = final_data[["punctuation_count_standardized", "comma_count", "exclamation_count", "female"]]
#X = final_data["female"]
y = final_data["generation"]
clf = LogisticRegression(random_state=0).fit(X, y)

#clf.predict(X[:2, :])
#clf.predict_proba(X[:2, :])
print(clf.score(X, y))
