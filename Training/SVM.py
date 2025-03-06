import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load dataset
df_svm = pd.read_csv(r".\Dataset\part3.csv")
X_svm = df_svm[['text', 'rating']]
y_svm = df_svm['label']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=45000, ngram_range=(1, 4)), 'text'),
        ('rating', StandardScaler(), ['rating'])
    ]
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
combined_predictions = np.zeros(len(y_svm)) 

# Train on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_svm, y_svm), 1):
    print(f"\nTraining SVM on Fold {fold}/6...")

    X_train, X_val = X_svm.iloc[train_idx], X_svm.iloc[val_idx]
    y_train, y_val = y_svm.iloc[train_idx], y_svm.iloc[val_idx]

    pipeline_svm = make_pipeline(preprocessor, SVC(C=3, class_weight='balanced', gamma='scale', kernel='rbf', probability=True, random_state=42))
    pipeline_svm.fit(X_train, y_train)

    # Get predicted probabilities for the positive class
    y_probs = pipeline_svm.predict_proba(X_val)[:, 1]

    # Calculate Mean Absolute Error for this fold
    fold_mae = np.mean(np.abs(y_val - y_probs)) 
    fold_maes.append(fold_mae)

    # Calculate custom accuracy for this fold
    fold_custom_accuracy = custom_accuracy(y_val, y_probs)
    fold_custom_accuracies.append(fold_custom_accuracy)

    # Calculate standard accuracy for this fold
    fold_predictions = (y_probs >= 0.5).astype(int) 
    fold_standard_accuracy = accuracy_score(y_val, fold_predictions)
    fold_standard_accuracies.append(fold_standard_accuracy)

    # Store the predictions to be averaged later
    combined_predictions[val_idx] = y_probs

    print(f"Fold {fold} MAE: {fold_mae:.4f}, Custom Accuracy: {fold_custom_accuracy:.4f}, Standard Accuracy: {fold_standard_accuracy:.4f}")

# Print Cross-Validation Results
print("\nSVM Cross-Validation Results:")
print(f"Mean MAE: {np.mean(fold_maes):.4f}")
print(f"Mean Custom Accuracy: {np.mean(fold_custom_accuracies):.4f}")
print(f"Mean Standard Accuracy: {np.mean(fold_standard_accuracies):.4f}")

# Train final SVM model on full dataset using combined K-Fold predictions
print("\nFinal training SVM on full dataset using combined K-Fold predictions...")
final_svm = make_pipeline(preprocessor, SVC(C=3, class_weight='balanced', gamma='scale', kernel='rbf', probability=True, random_state=42))
final_svm.fit(X_svm, y_svm)

# Save the final trained SVM model
joblib.dump(final_svm, r".\Models\svm_model.joblib")
print("SVM model saved as svm_mae_model.joblib.")
