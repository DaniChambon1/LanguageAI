from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC

def SVM(data):
    # Read in dataset with all features and drop unnecessary columns
    data = data.drop(["auhtor_ID","post","birth_year","language"], axis=1)
    # Define dependent and independent variable
    X = data.drop("Millennial", axis=1)
    y = data["Millennial"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline containing feature selection and hyper parameter tuning
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)), 
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    param_grid = {
        'feature_selection__k': [2,3,4,5,6,7],        # Number of features to select
        'svm__C': [0.1, 1, 10, 100],                  # Regularization parameter
        'svm__gamma': [1, 0.1, 0.01, 0.001],          # Kernel coefficient for 'rbf' kernel
        'svm__kernel': ['rbf', 'linear']              # Kernel type
    }

    # Implement GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Retrieve and print the selected features, and best parameters
    best_model = grid_search.best_estimator_
    selected_features = best_model.named_steps['feature_selection']
    selected_indices = selected_features.get_support(indices=True)
    feature_names = X.columns.tolist()
    selected_feature_names = [feature_names[i] for i in selected_indices]

    print("Selected Features SVM:", selected_feature_names)
    print("Best Parameters SVM:", grid_search.best_params_)

    # Evaluate the model with best parameters on the test set
    y_pred = best_model.predict(X_test)

    # Calculate accuracy, precision, recall, and F1-score for each class
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"SVM --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")