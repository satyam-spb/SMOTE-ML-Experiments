import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_random_forest(file_path, dataset_name):
    # Load dataset
    data = pd.read_csv(file_path)
    print(f"\n{'='*40}\nRandom Forest Evaluation on {dataset_name} Dataset\n{'='*40}")
    # Convert target to numeric codes
    data['Class'] = data['Class'].astype('category').cat.codes
    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Initialize and train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    # Predict on test set
    y_pred = rf_classifier.predict(X_test)
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=["normal", "Forwarding"])
    print(report)
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

# File paths
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate on original dataset
evaluate_random_forest(original_dataset, "Original")
# Evaluate on resampled dataset
evaluate_random_forest(resampled_dataset, "Resampled")
