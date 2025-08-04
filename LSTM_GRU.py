import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Embedding, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('resampled_dataset(3).csv')

# Encode the target labels
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])
num_classes = len(label_encoder.classes_)

# Separate features and target
X = data.drop(columns=['Class'])
y = to_categorical(data['Class'], num_classes)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for LSTM/GRU input (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build LSTM+GRU model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    Bidirectional(GRU(64, dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

# Make predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report (includes precision, recall, f1-score)
print("\nClassification Report:")
target_names = label_encoder.classes_
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
