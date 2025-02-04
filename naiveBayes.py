import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('resampled_dataset.csv')
# data = pd.read_csv('datasetSMOTE.csv')

# Optional: Inspect the first few rows and class distribution
print("Data Sample:")
print(data.head())
print("\nOriginal Class Distribution:")
print(data["Class"].value_counts())

# Convert the target to a categorical type and then to numerical codes
# This converts 'normal' to 0 and 'Forwarding' to 1 (alphabetical order)
data['Class'] = data['Class'].astype('category').cat.codes

# Separate features (all columns except 'Class') and target ('Class')
X = data.drop(columns=['Class'])
y = data['Class']

# Split data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Naïve Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Generate and print the classification report
report = classification_report(y_test, y_pred, target_names=["normal", "Forwarding"])
print("\nNaïve Bayes Classification Report:")
print(report)
