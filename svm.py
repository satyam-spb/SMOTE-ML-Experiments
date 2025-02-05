import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_svm(file_path, dataset_name):
    # Load the dataset
    data = pd.read_csv(file_path)
    print(f"\n{'='*40}\nSVM Evaluation on {dataset_name} Dataset\n{'='*40}")
    
    # Convert target to numerical codes (assumes "normal" and "Forwarding")
    data['Class'] = data['Class'].astype('category').cat.codes
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the SVM classifier
    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    
    # Print the evaluation results
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# File paths (all files in the same directory)
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate on both datasets
evaluate_svm(original_dataset, "Original")
evaluate_svm(resampled_dataset, "Resampled")
