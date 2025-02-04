import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = 'resampled_dataset.csv'
# data = pd.read_csv('datasetSMOTE.csv')
data = pd.read_csv(file_path)

# Optional: Inspect the first few rows and class distribution
print("Data Sample:")
print(data.head())
print("\nOriginal Class Distribution:")
print(data["Class"].value_counts())

# Convert the target column "Class" to a categorical type and then to numerical codes.
# This maps "normal" -> 0 and "Forwarding" -> 1 (alphabetical order).
data['Class'] = data['Class'].astype('category').cat.codes

# Separate features (all columns except 'Class') and target ('Class')
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Generate and print the classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred, target_names=["normal", "Forwarding"]))

# Compute and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Random Forest Confusion Matrix:")
print(cm)
