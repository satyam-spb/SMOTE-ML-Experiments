import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

def evaluate_ensemble(file_path, dataset_name):
    print(f"\n{'='*40}\nVoting Ensemble (DT + LR + NB) on {dataset_name} Dataset\n{'='*40}")

    # Load dataset
    data = pd.read_csv(file_path)

    # Convert target labels to numeric
    label_encoder = LabelEncoder()
    data['Class'] = label_encoder.fit_transform(data['Class'])

    # Features and labels
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define base classifiers
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=5000, random_state=42)
    nb = GaussianNB()

    # Voting Classifier (Hard Voting)
    ensemble_model = VotingClassifier(estimators=[
        ('Decision Tree', dt),
        ('Logistic Regression', lr),
        ('Naive Bayes', nb)
    ], voting='hard')  # Use 'soft' only if all support predict_proba

    # Train
    ensemble_model.fit(X_train, y_train)

    # Predict
    y_pred = ensemble_model.predict(X_test)

    # Decode labels for readability
    target_names = label_encoder.classes_

    # Evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# File path
dataset_path = 'resampled_dataset(3).csv'
evaluate_ensemble(dataset_path, "Resampled")
