#from feature_hilde import combined_gen
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from imblearn.over_sampling import SMOTE

combined_gen = pd.read_csv("data\combined_gen.csv")
columns_to_exclude = ['birth_year','language','post','Unnamed: 0','auhtor_ID']
combined_gen2 = combined_gen.copy().drop(columns=columns_to_exclude)


sampled = combined_gen2.sample(n=2000, random_state=42).reset_index(drop=True)
X = sampled.drop("Millennial",axis=1)
y = sampled["Millennial"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)), 
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'feature_selection__k': [2,3,4,5],            # Number of features to select
    'svm__C': [0.1, 1, 10, 100],                  # Regularization parameter
    'svm__gamma': [1, 0.1, 0.01, 0.001],          # Kernel coefficient for 'rbf' kernel
    'svm__kernel': ['rbf', 'linear']              # Kernel type
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)
grid_search.fit(X_resampled, y_resampled)


best_model = grid_search.best_estimator_
selected_features = best_model.named_steps['feature_selection']
selected_indices = selected_features.get_support(indices=True)
feature_names = X_resampled.columns.tolist()
selected_feature_names = [feature_names[i] for i in selected_indices]


print("Selected Features:", selected_feature_names)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


chosen_kernel = grid_search.best_params_['svm__kernel']
chosen_C = grid_search.best_params_['svm__C']
chosen_gamma = grid_search.best_params_['svm__gamma']

#### Performing cross validation on entire dataset with chosen parameters

X = combined_gen2.drop("Millennial",axis=1)
y = combined_gen2["Millennial"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled[selected_feature_names], y_resampled, test_size=0.2, random_state=42)
model = svm.SVC(kernel=chosen_kernel, C = chosen_C, gamma = chosen_gamma)
num_folds = 5


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
# Create a k-fold cross-validation iterator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_resampled):
    X_train, X_test = X_resampled.iloc[train_index][selected_feature_names], X_resampled.iloc[test_index][selected_feature_names]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Make predictions on the test fold
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

counter = 0 
for i in predictions:
    if i == 0:
        counter += 1
print(counter)




# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# predictions_train = model.predict(X_train)
# accuracy = accuracy_score(y_test, predictions)
# precision = precision_score(y_test, predictions)
# recall = recall_score(y_test, predictions)
# f1_test = f1_score(y_test, predictions)
# f1_train = f1_score(y_train, predictions_train)
# print(f"SVM --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score (train): {f1_train}, F1-score (test): {f1_test}")

# print(sum(predictions_train))
# print(sum(predictions))
# X_entire.shape()
