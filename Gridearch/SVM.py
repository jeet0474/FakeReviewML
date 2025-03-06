import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load dataset
df_svm = pd.read_csv(r".\Dataset\part3.csv")
X_svm = df_svm[['text', 'rating']]
y_svm = df_svm['label']

# Define preprocessor: we use ColumnTransformer for text and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text'),
        ('rating', StandardScaler(), ['rating']) 
    ]
)

# Define parameter grid for SVM and TfidfVectorizer
param_grid = {
    # TfidfVectorizer parameters
    'preprocessor__text__max_features': [40000, 45000, 50000], 
    'preprocessor__text__ngram_range': [(1, 4), (1, 5), (1, 6)],
    
    # SVM parameters
    'svc__C': [ 1.0, 3.0, 4, 5, 6, 7, 8,], 
    'svc__kernel': ['linear', 'poly', 'rbf',], 
    'svc__class_weight': ['balanced'],
    'svc__gamma': ['scale', 'auto'],  
    'svc__probability': [True]
}

# Create the pipeline
pipeline_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('svc', SVC(random_state=42)) 
])

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline_svm, param_grid, cv=6, n_jobs=-1, verbose=2, scoring='accuracy')

# Train the model using GridSearchCV
print("Starting GridSearchCV for SVM...")
grid_search.fit(X_svm, y_svm)

# Get the best parameters and the best model
print(f"\nBest parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Cross-validation results from GridSearchCV
print("\nGridSearchCV Results:")
cv_results = pd.DataFrame(grid_search.cv_results_)
print(cv_results[['param_preprocessor__text__max_features', 
                  'param_preprocessor__text__ngram_range', 
                  'param_svc__C', 'param_svc__kernel', 'mean_test_score', 'std_test_score']])

# Test the best model on the validation data
print("\nEvaluating the best model on the test data...")

# Assuming the dataset has a test set, use the best model to predict and evaluate performance
df_test = pd.read_csv(r".\Dataset\test.csv")
X_test = df_test[['text', 'rating']]
y_test = df_test['label']

# Get predictions and evaluate
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

# Custom accuracy function
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

# Evaluate standard accuracy
standard_accuracy = accuracy_score(y_test, y_pred)

# Calculate custom accuracy
custom_acc = custom_accuracy(y_test, y_probs)

# Print evaluation results
print(f"Standard Accuracy: {standard_accuracy:.4f}")
print(f"Custom Accuracy: {custom_acc:.4f}")
