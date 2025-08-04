import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def evaluate_svm(file_path, dataset_name):
    print(f"\n{'='*40}\nSVM Evaluation on {dataset_name} Dataset\n{'='*40}")
    
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Convert 'Class' column to numeric codes
    data['Class'] = data['Class'].astype('category').cat.codes
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale + Train using pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42, class_weight='balanced', kernel='linear'))
    ])

    print("Starting SVM model training...")
    pipeline.fit(X_train, y_train)
    print("SVM model training completed.")
    
    y_pred = pipeline.predict(X_test)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"], zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Extract & return metrics
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

# Run for both datasets
evaluate_svm(original_dataset, "Original")
evaluate_svm(resampled_dataset, "Resampled")
