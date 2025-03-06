import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define a preprocessor that handles both text and rating features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text'), 
        ('rating', StandardScaler(), ['rating'])
    ]
)

# Define the function to perform GridSearchCV and evaluate on the test dataset
def perform_grid_search_and_test(model, file_path, model_name, param_grid, test_file_path, n_splits=5):
    # Load the training dataset
    df_train = pd.read_csv(file_path)
    X_train = df_train[['text', 'rating']]
    y_train = df_train['label']

    # Load the test dataset
    df_test = pd.read_csv(test_file_path)
    X_test = df_test[['text', 'rating']] 
    y_test = df_test['label'] 

    # Create a pipeline with preprocessing and the model
    pipeline = make_pipeline(preprocessor, model) 

    # Set up GridSearchCV with cross-validation (but we won't use it for training, only for hyperparameter search)
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid,
        cv=n_splits,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    # Train with GridSearchCV on the training dataset
    print(f"\nTraining {model_name} with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Test the best model on the test dataset
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Print the best parameters and the test accuracy
    print(f"\nBest Hyperparameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Define the updated hyperparameter grid for Random Forest and TfidfVectorizer
param_grid = {
    'columntransformer__text__max_features': [20000],
    'columntransformer__text__ngram_range': [(1, 2), (1, 3)], 
    'randomforestclassifier__n_estimators': [1100, 1200, 1300], 
    'randomforestclassifier__max_depth': [80, 90, 100, 110], 
    'randomforestclassifier__min_samples_split': [5], 
    'randomforestclassifier__min_samples_leaf': [2], 
    'randomforestclassifier__max_features': ['sqrt'], 
    'randomforestclassifier__bootstrap': [False],
    'randomforestclassifier__class_weight': ['balanced']
}

# Perform GridSearchCV and test Random Forest on the test dataset
perform_grid_search_and_test(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    r".\Dataset\part2.csv",
    "Random Forest",
    param_grid, 
    r".\Dataset\test.csv" 
)
