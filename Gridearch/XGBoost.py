import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load trained models
logistic_model = joblib.load(r".\Models\logistic_regression_model.joblib")
rf_model = joblib.load(r".\Models\random_forest_model.joblib")
svm_model = joblib.load(r".\Models\svm_model.joblib")

# Load dataset
df_stack = pd.read_csv(r".\Dataset\part4.csv")
df_stack = df_stack.dropna()

# Extract features for stacking
X_features = df_stack[['text', 'rating']]
y_stack = df_stack['label']

# Get probability predictions from trained models
df_stack['logistic_pred'] = logistic_model.predict_proba(X_features)[:, 1]
df_stack['rf_pred'] = rf_model.predict_proba(X_features)[:, 1]
df_stack['svm_pred'] = svm_model.predict_proba(X_features)[:, 1]

# Select features for stacking model
X_stack = df_stack[['logistic_pred', 'rf_pred', 'svm_pred']]

# Define custom accuracy calculation function
def custom_accuracy(y_true, y_pred_probs):
    """
    Custom accuracy:
    - If label is 1, take the probability as is.
    - If label is 0, take 1 - probability.
    If label is 1 and probability is 0.9, accuracy is 0.9.
    If label is 0 and probability is 0.9, accuracy is 0.1 because we want to be as far as possible from 1 if the text is authentic.
    """
    return np.mean([prob if true == 1 else (1 - prob) for true, prob in zip(y_true, y_pred_probs)])

# Create a scorer for custom accuracy to use in GridSearchCV
custom_scorer = make_scorer(custom_accuracy, greater_is_better=True)

param_grid = {
    'learning_rate': [0.0025, 0.005, 0.0075, 0.01, 0.05, 0.015, 0.1],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': [400, 500, 800, 1000, 1100, 1200, 1300, 1400, 1500],
    'subsample': [0.85, 0.9, 0.92, 0.94, 0.95, 0.97, 0.98],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Define XGBoost model
stack_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Perform GridSearchCV with StratifiedKFold cross-validation
kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=stack_model,
    param_grid=param_grid,
    cv=kf,
    scoring=custom_scorer,
    n_jobs=-1,
    verbose=2
)

# Train with GridSearchCV
print("\nPerforming Grid Search for Stacking Model...")
grid_search.fit(X_stack, y_stack)

# Print best parameters only
print("\nBest Hyperparameters for Stacking Model:")
print(grid_search.best_params_)

print(f"\nBest Custom Accuracy Achieved: {grid_search.best_score_:.4f}")