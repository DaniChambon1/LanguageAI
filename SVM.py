from feature_hilde import combined_gen
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = combined_gen[["capital count (stand.)","emoticon count (stand.)","pronoun count (stand.)", "female"]][:2000]
y = combined_gen["Millennial"][:2000]


pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)), 
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'feature_selection__k': [2,3,4],                  # Number of features to select
    'svm__C': [0.1, 1, 10, 100],                  # Regularization parameter
    'svm__gamma': [1, 0.1, 0.01, 0.001],          # Kernel coefficient for 'rbf' kernel
    'svm__kernel': ['rbf', 'linear']              # Kernel type
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)
grid_search.fit(X, y)


best_model = grid_search.best_estimator_
selected_features = best_model.named_steps['feature_selection']
selected_indices = selected_features.get_support(indices=True)
feature_names = ["capital count (stand.)","emoticon count (stand.)","pronoun count (stand.)", "female"]
selected_feature_names = [feature_names[i] for i in selected_indices]

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Selected Features:", selected_feature_names)
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = svm.SVC(kernel='linear')
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# precision = precision_score(y_test, predictions)
# recall = recall_score(y_test, predictions)
# f1 = f1_score(y_test, predictions)
# print(f"SVM --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")