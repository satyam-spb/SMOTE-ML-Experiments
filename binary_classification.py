import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_binary_classification(file_path, dataset_name):
    # Load the dataset
    data = pd.read_csv(file_path)
    print(f"\n{'='*40}\nBinary Classification (Voting) on {dataset_name} Dataset\n{'='*40}")
    
    # Convert target to numerical codes (assumes "normal" and "Forwarding")
    data['Class'] = data['Class'].astype('category').cat.codes
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize individual classifiers
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = SVC(probability=True, random_state=42)
    
    # Create the Voting Classifier (soft voting)
    voting_clf = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2)], voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    
    # Print the evaluation results
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# File paths
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate on both datasets
evaluate_binary_classification(original_dataset, "Original")
evaluate_binary_classification(resampled_dataset, "Resampled")
