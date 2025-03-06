import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import StratifiedKFold
import time

def load_model(model_path, model_name):
    start_time = time.time()
    model = joblib.load(model_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{model_name} loaded successfully in {elapsed_time:.4f} seconds")
    return model

logistic_model = load_model(r".\Models\logistic_regression_model.joblib", "Logistic Regression Model")
rf_model = load_model(r".\Models\random_forest_model.joblib", "Random Forest Model")
svm_model = load_model(r".\Models\svm_model.joblib", "SVM Model")

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

# Define stacking model
stack_model = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.0075,
    max_depth=4,
    subsample=0.98,
    colsample_bytree=1,
    n_estimators=800,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Custom accuracy calculation function
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

# Define K-Fold
kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
fold_maes = []
fold_custom_accuracies = [] 
fold_standard_accuracies = [] 
combined_predictions = np.zeros(len(y_stack))

# Train on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_stack, y_stack), 1):
    print(f"\nTraining Stacking Model on Fold {fold}/5...")

    X_train, X_val = X_stack.iloc[train_idx], X_stack.iloc[val_idx]
    y_train, y_val = y_stack.iloc[train_idx], y_stack.iloc[val_idx]

    # Fit the stacking model
    stack_model.fit(X_train, y_train)

    # Get predicted probabilities for the positive class
    y_probs = stack_model.predict_proba(X_val)[:, 1]

    # Calculate Mean Absolute Error (MAE) for this fold
    fold_mae = mean_absolute_error(y_val, y_probs)  # Calculate MAE
    fold_maes.append(fold_mae)

    # Calculate custom accuracy for this fold
    fold_custom_accuracy = custom_accuracy(y_val, y_probs)
    fold_custom_accuracies.append(fold_custom_accuracy)

    # Calculate standard accuracy for this fold
    fold_predictions = (y_probs >= 0.5).astype(int) 
    fold_standard_accuracy = accuracy_score(y_val, fold_predictions)
    fold_standard_accuracies.append(fold_standard_accuracy)

    # Store the predictions to be averaged later (optional, not used for final training)
    combined_predictions[val_idx] = y_probs

    print(f"Fold {fold} MAE: {fold_mae:.4f}, Custom Accuracy: {fold_custom_accuracy:.4f}, Standard Accuracy: {fold_standard_accuracy:.4f}")

# Print Cross-Validation Results
print("\nStacking Model Cross-Validation Results:")
print(f"Mean MAE: {np.mean(fold_maes):.4f}")
print(f"Mean Custom Accuracy: {np.mean(fold_custom_accuracies):.4f}")
print(f"Mean Standard Accuracy: {np.mean(fold_standard_accuracies):.4f}")

# Final Training on Full Data
print("\nFinal training Stacking Model on full dataset...")
stack_model.fit(X_stack, y_stack)

# Save Stacking Model
joblib.dump(stack_model, r".\Models\stacking_model.joblib")
print("Stacking model saved as stacking_model_mae.joblib.")
