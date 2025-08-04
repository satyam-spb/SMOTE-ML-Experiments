import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_decision_tree(file_path, dataset_name):
    print(f"\n{'='*40}\nDecision Tree Evaluation on {dataset_name} Dataset\n{'='*40}")
    
    # Load dataset
    data = pd.read_csv(file_path)

    # Convert target to numeric codes
    data['Class'] = data['Class'].astype('category').cat.codes
    
    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Decision Tree model
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = dt_classifier.predict(X_test)

    # Compute and print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"], zero_division=0))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Compute individual metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Return values if needed later
    return {"dataset": dataset_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# File paths
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate both datasets
evaluate_decision_tree(original_dataset, "Original")
evaluate_decision_tree(resampled_dataset, "Resampled")
