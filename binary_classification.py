import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def evaluate_binary_classification(file_path, dataset_name):
    print(f"\n{'='*40}\nBinary Classification (Voting) on {dataset_name} Dataset\n{'='*40}")
    data = pd.read_csv(file_path)
    
    # Convert target to numerical codes (assumes "normal" and "Forwarding")
    data['Class'] = data['Class'].astype('category').cat.codes
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Optional: Use a subset for faster debugging if needed
    # X, y = X.sample(frac=0.1, random_state=42), y.sample(frac=0.1, random_state=42)
    
    # Scale the data using a pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Use Logistic Regression with increased max_iter and SVM with a linear kernel for faster training
    clf1 = LogisticRegression(random_state=42, max_iter=5000)
    clf2 = SVC(probability=True, random_state=42, kernel='linear')
    
    # Create the Voting Classifier (soft voting)
    voting_clf = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2)], voting='soft')
    
    print("Starting Voting Classifier training...")
    voting_clf.fit(X_train, y_train)
    print("Voting Classifier training completed.")
    
    y_pred = voting_clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# File paths (all files in the same directory)
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate on both datasets
evaluate_binary_classification(original_dataset, "Original")
evaluate_binary_classification(resampled_dataset, "Resampled")
