import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def evaluate_confusion_matrix(file_path, dataset_name):
    # Load dataset
    data = pd.read_csv(file_path)
    print(f"\n{'='*40}\nConfusion Matrix for {dataset_name} Dataset\n{'='*40}")
    # Convert target to numeric codes
    data['Class'] = data['Class'].astype('category').cat.codes
    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Train the Naïve Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    # Make predictions
    y_pred = nb_classifier.predict(X_test)
    # Compute and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

# File paths
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate on original dataset
evaluate_confusion_matrix(original_dataset, "Original")
# Evaluate on resampled dataset
evaluate_confusion_matrix(resampled_dataset, "Resampled")
