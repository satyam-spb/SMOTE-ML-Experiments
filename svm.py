import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def evaluate_svm(file_path, dataset_name):
    print(f"\n{'='*40}\nSVM Evaluation on {dataset_name} Dataset\n{'='*40}")
    data = pd.read_csv(file_path)
    
    # Convert target to numerical codes (assumes "normal" and "Forwarding")
    data['Class'] = data['Class'].astype('category').cat.codes
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Optional: Use only a subset for testing if the dataset is huge
    # X, y = X.sample(frac=0.1, random_state=42), y.sample(frac=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Use a pipeline to scale data and use a linear kernel for speed
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42, class_weight='balanced', kernel='linear'))
    ])
    
    print("Starting SVM model training...")
    pipeline.fit(X_train, y_train)
    print("SVM model training completed.")
    
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# File paths (all files in the same directory)
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

# Evaluate on both datasets
evaluate_svm(original_dataset, "Original")
evaluate_svm(resampled_dataset, "Resampled")
