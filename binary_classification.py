import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler

def evaluate_binary_classification(file_path, dataset_name):
    print(f"\n{'='*40}\nBinary Classification (Voting) on {dataset_name} Dataset\n{'='*40}")
    
    # Load data
    data = pd.read_csv(file_path)

    # Encode target class to numeric
    data['Class'] = data['Class'].astype('category').cat.codes
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Define base classifiers
    clf1 = LogisticRegression(random_state=42, max_iter=5000)
    clf2 = SVC(probability=True, random_state=42, kernel='linear')

    # Ensemble: Voting Classifier
    voting_clf = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2)], voting='soft')

    # Train the model
    print("Starting Voting Classifier training...")
    voting_clf.fit(X_train, y_train)
    print("Voting Classifier training completed.")

    # Predict
    y_pred = voting_clf.predict(X_test)

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"], zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nIndividual Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

# Run on both datasets
evaluate_binary_classification('datasetSMOTE.csv', "Original")
evaluate_binary_classification('resampled_dataset.csv', "Resampled")
