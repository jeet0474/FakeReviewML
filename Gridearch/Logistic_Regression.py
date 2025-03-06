from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score

# Define a preprocessor that handles both text and rating features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text'),
        ('rating', StandardScaler(), ['rating'])
    ]
)

# Custom accuracy calculation
def custom_accuracy(y_true, y_pred_probs):
    """
    Custom accuracy:
    - If label is 1, take the probability as is.
    - If label is 0, take 1 - probability.
    If label is 1 and probability is 0.9, accuracy is 0.9.
    If label is 0 and probability is 0.9, accuracy is 0.1 because we want to be as far as possible from 1 if the text is authentic.
    """
    custom_acc = 0
    for true_label, pred_prob in zip(y_true, y_pred_probs):
        if true_label == 1:
            custom_acc += pred_prob
        else:
            custom_acc += (1 - pred_prob)
    return custom_acc / len(y_true)

# Train with GridSearchCV and K-Fold to find best Logistic Regression parameters
def train_with_grid_search_and_kfold(model, train_file_path, test_file_path, param_grid, n_splits=2):
    # Load the training dataset
    df_train = pd.read_csv(train_file_path)
    X_train = df_train[['text', 'rating']]
    y_train = df_train['label'] 
    
    # Load the test dataset
    df_test = pd.read_csv(test_file_path)
    X_test = df_test[['text', 'rating']]
    y_test = df_test['label'] 

    # Create a pipeline with preprocessing and model
    pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(random_state=42)
    )

    # Set up GridSearchCV without n_jobs
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid,
        cv=n_splits,
        verbose=2,
        scoring='accuracy'
    )

    # Train with GridSearchCV on the training dataset
    print(f"\nTraining with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model from GridSearchCV
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate on the test dataset
    y_test_pred_probs = best_model.predict_proba(X_test)[:, 1]
    best_test_custom_accuracy = custom_accuracy(y_test, y_test_pred_probs)
    best_test_predictions = (y_test_pred_probs >= 0.5).astype(int)
    best_test_standard_accuracy = accuracy_score(y_test, best_test_predictions)

    print(f"\nBest Parameters: {best_params}")
    print(f"Test Set - Custom Accuracy: {best_test_custom_accuracy:.4f}")
    print(f"Test Set - Standard Accuracy: {best_test_standard_accuracy:.4f}")

# Define the hyperparameter grid for Logistic Regression and TfidfVectorizer
param_grid = {
    'logisticregression__C': [28, 30, 32, 34],
    'logisticregression__max_iter': [400, 300, 500], 
    'logisticregression__solver': ['lbfgs'],
    'logisticregression__class_weight': ['balanced'], 
    'logisticregression__tol': [1e-3, 1e-4], 
    'columntransformer__text__max_features': [40000, 45000, 50000, 55000], 
    'columntransformer__text__ngram_range': [(1, 3), (1, 4), (1, 5)] 
}


# Train Logistic Regression with GridSearchCV and K-Fold combined learning
train_with_grid_search_and_kfold(
    LogisticRegression(random_state=42), 
    r".\Dataset\part1.csv", 
    r".\Dataset\test.csv",
    param_grid,
    n_splits=6
)
