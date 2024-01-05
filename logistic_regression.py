#from feature_hilde import balanced_gen
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,  KFold
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data/balanced_features.csv")
data = data.drop(["Unnamed: 0.2","Unnamed: 0.1","Unnamed: 0","auhtor_ID","post","birth_year","language"],axis=1)

sampled = data.sample(n=2000, random_state=42).reset_index(drop=True)
X = sampled.drop("Millennial", axis=1)
y = sampled["Millennial"]

# #
pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)), 
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression())
])

param_grid = {
    'feature_selection__k': [2,3,4,5],          # Number of features to select
    'logistic__penalty': ['l1', 'l2'],          # Specify the norm of the penalty:
    'logistic__C': np.logspace(-4, 4, 20),      # Inverse of regularization strength
    'logistic__solver': ['liblinear']           # Algorithm to use in the optimization problem
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)
grid_search.fit(X, y)


best_model = grid_search.best_estimator_
selected_features = best_model.named_steps['feature_selection']
selected_indices = selected_features.get_support(indices=True)
feature_names = X.columns.tolist()
selected_feature_names = [feature_names[i] for i in selected_indices]


print("Selected Features:", selected_feature_names)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


chosen_penalty = grid_search.best_params_['logistic__penalty']
chosen_C = grid_search.best_params_['logistic__C']
chosen_solver = grid_search.best_params_['logistic__solver']

#### Performing cross validation on entire dataset with chosen parameters

X = data.drop("Millennial", axis=1)
y = data["Millennial"]

X_train, X_test, y_train, y_test = train_test_split(X[selected_feature_names], y, test_size=0.2, random_state=42)
model = LogisticRegression(penalty=chosen_penalty, C=chosen_C, solver=chosen_solver)
num_folds = 5

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index][selected_feature_names], X.iloc[test_index][selected_feature_names]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate evaluation metrics for this fold
    accuracy_scores.append(accuracy_score(y_test, predictions))
    precision_scores.append(precision_score(y_test, predictions))
    recall_scores.append(recall_score(y_test, predictions))
    f1_scores.append(f1_score(y_test, predictions))

# Compute average scores across all folds
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)

# Print average evaluation metrics across folds
print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1-score: {avg_f1}")










# from sklearn.linear_model import LogisticRegression
# from feature_hilde import combined_gen
# from sklearn.model_selection import train_test_split, GridSearchCV,  KFold
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# #takes 2000 rows of the dataset to do the crossvalidation + hyperparameter tuning over. 
# columns_to_exclude = ['Millennial', 'birth_year','language','post','Unnamed: 0','auhtor_ID']
# combined_gen2 = combined_gen.copy().drop(columns=columns_to_exclude)

# X = combined_gen2.iloc[:2000]
# y = combined_gen["Millennial"][:2000]


# pipeline = Pipeline([
#     ('feature_selection', SelectKBest(score_func=f_classif)), 
#     ('scaler', StandardScaler()),
#     ('logistic', LogisticRegression())
# ])

# param_grid = {
#     'feature_selection__k': [2,3,4,5],              # Number of features to select
#     'logistic__penalty': ['l1', 'l2', 'elasticnet', 'none'],                        # Specify the norm of the penalty:
#     'logistic__C': [0.01, 0.1, 1, 10],                                              # Inverse of regularization strength
#     'logistic__solver': ['lbfgs', 'liblinear', 'newton-c', 'newton-cholesky', 'sag'] # Algorithm to use in the optimization problem
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)
# grid_search.fit(X, y)


# best_model = grid_search.best_estimator_
# selected_features = best_model.named_steps['feature_selection']
# selected_indices = selected_features.get_support(indices=True)
# feature_names = X.columns.tolist()
# selected_feature_names = [feature_names[i] for i in selected_indices]

# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Selected Features:", selected_feature_names)
# print("Best Parameters:", best_params)
# print("Best Score:", best_score)


# chosen_penalty = best_params['logistic__penalty']
# chosen_C = best_params['logistic__C']
# chosen_solver = best_params['logistic__solver']

# #trains selected model from above on the whole dataset

# X_entire = combined_gen2
# y_entire = combined_gen["Millennial"]
# X_train, X_test, y_train, y_test = train_test_split(X_entire[selected_feature_names], y_entire, test_size=0.2, random_state=42)
# model = LogisticRegression(penalty=chosen_penalty, C=chosen_C, solver=chosen_solver)
# num_folds = 5
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []
# # Create a k-fold cross-validation iterator
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# for train_index, test_index in kf.split(X_entire):
#     X_train, X_test = X_entire.iloc[train_index][selected_feature_names], X_entire.iloc[test_index][selected_feature_names]
#     y_train, y_test = y_entire.iloc[train_index], y_entire.iloc[test_index]

#     # Fit the model on training data
#     model.fit(X_train, y_train)

#     # Make predictions on the test fold
#     predictions = model.predict(X_test)

#     # Calculate evaluation metrics for this fold
#     accuracy_scores.append(accuracy_score(y_test, predictions))
#     precision_scores.append(precision_score(y_test, predictions))
#     recall_scores.append(recall_score(y_test, predictions))
#     f1_scores.append(f1_score(y_test, predictions))

# # Compute average scores across all folds
# avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
# avg_precision = sum(precision_scores) / len(precision_scores)
# avg_recall = sum(recall_scores) / len(recall_scores)
# avg_f1 = sum(f1_scores) / len(f1_scores)

# # Print average evaluation metrics across folds
# print(f"Average Accuracy: {avg_accuracy}")
# print(f"Average Precision: {avg_precision}")
# print(f"Average Recall: {avg_recall}")
# print(f"Average F1-score: {avg_f1}")
