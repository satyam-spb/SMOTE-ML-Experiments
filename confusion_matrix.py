import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv('resampled_dataset.csv')
# data = pd.read_csv('datasetSMOTE.csv')

# Inspect the first few rows and class distribution (optional)
print("Data Sample:")
print(data.head())
print("\nOriginal Class Distribution:")
print(data["Class"].value_counts())

# Convert the target column to categorical and then numerical codes
data['Class'] = data['Class'].astype('category').cat.codes

# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naïve Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix clearly
print("\nConfusion Matrix:")
print(cm)
