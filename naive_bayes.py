import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_naive_bayes(file_path, dataset_name):
    # Load dataset
    data = pd.read_csv(file_path)
    print(f"\n{'='*40}\nNaïve Bayes Evaluation on {dataset_name} Dataset\n{'='*40}")
    
    # Convert the target column to categorical and then to numerical codes
    data['Class'] = data['Class'].astype('category').cat.codes

    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Naïve Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = nb_classifier.predict(X_test)

    # Calculate and print evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # You can also return these metrics for storing in a table or plot later
    return acc, prec, rec, f1

# File paths
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate on both datasets
evaluate_naive_bayes(original_dataset, "Original")
evaluate_naive_bayes(resampled_dataset, "Resampled")
