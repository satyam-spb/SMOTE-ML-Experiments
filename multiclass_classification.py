import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_multiclass_classification(file_path, dataset_name):
    # Load the dataset
    data = pd.read_csv(file_path)
    print(f"\n{'='*40}\nMulticlass Classification Evaluation on {dataset_name} Dataset\n{'='*40}")
    
    # Create a new multiclass target by splitting "normal" into "normal1" and "normal2"
    def assign_multiclass(label):
        label = label.strip().lower()
        if label == "normal":
            # Randomly assign one of the two subcategories
            return np.random.choice(["normal1", "normal2"])
        else:
            return label  # "forwarding" remains unchanged

    data['MultiClass'] = data['Class'].apply(assign_multiclass)
    print("New class distribution for multiclass:")
    print(data['MultiClass'].value_counts())
    
    # Convert the new multiclass target to string and then to categorical to ensure proper mapping
    data['MultiClass'] = data['MultiClass'].astype(str).astype('category')
    mapping = dict(enumerate(data['MultiClass'].cat.categories))
    print("Mapping of numeric codes to classes:", mapping)
    data['MultiClass_Code'] = data['MultiClass'].cat.codes
    
    # Define features and target (dropping the original 'Class' and temporary 'MultiClass' columns)
    X = data.drop(columns=['Class', 'MultiClass', 'MultiClass_Code'])
    y = data['MultiClass_Code']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Logistic Regression for multiclass classification using lbfgs solver.
    # Note: The multi_class parameter is left to its default so that it automatically selects multinomial if needed.
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    lr_classifier.fit(X_train, y_train)
    y_pred = lr_classifier.predict(X_test)
    
    # Prepare target names based on the mapping dictionary
    target_names = [mapping[i] for i in range(len(mapping))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# File paths (all files are in the same directory)
original_dataset = 'datasetSMOTE.csv'
resampled_dataset = 'resampled_dataset.csv'

evaluate_multiclass_classification(original_dataset, "Original")
evaluate_multiclass_classification(resampled_dataset, "Resampled")
