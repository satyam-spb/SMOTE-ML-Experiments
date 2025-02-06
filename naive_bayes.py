import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

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
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=["normal", "Forwarding"])
    print(report)

# File paths
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'
# Evaluate on original dataset
evaluate_naive_bayes(original_dataset, "Original")
# Evaluate on resampled dataset
evaluate_naive_bayes(resampled_dataset, "Resampled")
