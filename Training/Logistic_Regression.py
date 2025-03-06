import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import os

# Define a preprocessor that handles both text and rating features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=50000, ngram_range=(1, 4)), 'text'),
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

# Train the model with K-Fold and compute MAE, custom accuracy, and standard accuracy
def train_with_combined_kfold_mae(model, file_path, model_name, n_splits=6):
    # Load the dataset
    df = pd.read_csv(file_path)
    X = df[['text', 'rating']] 
    y = df['label'] 

    # Convert X and y to DataFrame/Series (for StratifiedKFold and splitting)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_maes = []
    fold_custom_accuracies = []
    fold_standard_accuracies = []
    combined_predictions = np.zeros(len(y)) 

    # Perform K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining {model_name} on Fold {fold + 1}/{n_splits}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create a pipeline with preprocessing and model
        pipeline = make_pipeline(preprocessor, model)

        # Train on the training fold
        pipeline.fit(X_train, y_train)

        # Predict probabilities on validation fold
        val_probs = pipeline.predict_proba(X_val)[:, 1]

        # Store predictions to be averaged later
        combined_predictions[val_idx] = val_probs

        # Calculate Mean Absolute Error for this fold
        fold_mae = np.mean(np.abs(y_val - val_probs)) 
        fold_maes.append(fold_mae)

        # Calculate custom accuracy for this fold
        fold_custom_accuracy = custom_accuracy(y_val, val_probs)
        fold_custom_accuracies.append(fold_custom_accuracy)

        # Calculate standard accuracy for this fold
        fold_predictions = (val_probs >= 0.5).astype(int) 
        fold_standard_accuracy = accuracy_score(y_val, fold_predictions)
        fold_standard_accuracies.append(fold_standard_accuracy)

        print(f"Fold {fold + 1} MAE: {fold_mae:.4f}, Custom Accuracy: {fold_custom_accuracy:.4f}, Standard Accuracy: {fold_standard_accuracy:.4f}")

    # Print average results across all folds
    print(f"\n{model_name} Cross-Validation Results:")
    print(f"Mean MAE: {np.mean(fold_maes):.4f}")
    print(f"Mean Custom Accuracy: {np.mean(fold_custom_accuracies):.4f}")
    print(f"Mean Standard Accuracy: {np.mean(fold_standard_accuracies):.4f}")

    # Train final model using the combined K-Fold predictions
    print(f"\nFinal training {model_name} on all data using K-Fold predictions...")
    final_pipeline = make_pipeline(preprocessor, model)
    final_pipeline.fit(X, y) 

    # Save the final trained model
    model_filename = os.path.join(r".\Models", f"{model_name.lower().replace(' ', '_')}_model.joblib")
    joblib.dump(final_pipeline, model_filename)
    print(f"{model_name} model saved as {model_filename}.")

# Train Logistic Regression with K-Fold combined learning and custom accuracy
train_with_combined_kfold_mae(
    LogisticRegression( 
        C=30,
        max_iter=400,
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=-1,
        tol=1e-3,
        random_state=42
    ), r".\Dataset\part1.csv", "Logistic Regression")
