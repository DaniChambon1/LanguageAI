from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

balanced_gen = pd.read_csv("data/balanced_gen_plus.csv")
balanced_gen2 = balanced_gen[['Millennial','contraction count','exaggeration count','capital count','emoticon count','pronoun count','punctuation count','comma count','exclamation count','female']].copy()

X = balanced_gen2.drop("Millennial", axis=1)
y = balanced_gen2["Millennial"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline containing feature selection and hyper parameter tuning
pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)), 
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression())
])

param_grid = {
    'feature_selection__k': [2,3,4,5,6],                # Number of features to select
    'logistic__penalty': ['l1', 'l2'],                  # Specify the norm of the penalty
    'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100],      # Inverse of regularization strength
    'logistic__solver': ['liblinear','sag']             # Algorithm to use in the optimization problem
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
selected_features = best_model.named_steps['feature_selection']
selected_indices = selected_features.get_support(indices=True)
feature_names = X.columns.tolist()
selected_feature_names = [feature_names[i] for i in selected_indices]


print("Selected Features:", selected_feature_names)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the model with best parameters on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))