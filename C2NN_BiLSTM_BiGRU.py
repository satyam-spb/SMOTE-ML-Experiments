import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv('resampled_dataset(3).csv')

# Encode labels
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])
num_classes = len(label_encoder.classes_)

# Features and labels
X = data.drop(columns=['Class'])
y = to_categorical(data['Class'], num_classes)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for CNN/LSTM/GRU
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build CNN + LSTM + GRU model
model = Sequential([
    Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=1),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    Bidirectional(GRU(64, dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

# Predictions and metrics
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Precision, Recall, F1
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"Precision (weighted): {prec:.4f}")
print(f"Recall (weighted):    {rec:.4f}")
print(f"F1 Score (weighted):  {f1:.4f}")
