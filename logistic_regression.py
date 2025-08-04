import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

def evaluate_logistic_regression(file_path, dataset_name):
    print(f"\n{'='*40}\nLogistic Regression Evaluation on {dataset_name} Dataset\n{'='*40}")
    
    # Load the dataset
    data = pd.read_csv(file_path)

    # Encode the target column
    data['Class'] = data['Class'].astype('category').cat.codes
    
    # Split features and target
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train Logistic Regression
    lr_classifier = LogisticRegression(random_state=42, max_iter=5000)
    lr_classifier.fit(X_train, y_train)
    
    # Predict test labels
    y_pred = lr_classifier.predict(X_test)
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"], zero_division=0))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Individual Metric Values
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {"dataset": dataset_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# File paths
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Run evaluations
evaluate_logistic_regression(original_dataset, "Original")
evaluate_logistic_regression(resampled_dataset, "Resampled")
