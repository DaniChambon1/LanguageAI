from feature_hilde import combined_gen
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

X = combined_gen[["capital count","capital count (stand.)","emoticon count","emoticon count (stand.)","pronoun count","pronoun count (stand.)", "female"]]
y = combined_gen["Millennial"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(f"kNN --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")