import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv('resampled_dataset(3).csv')

# Encode class labels
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])
num_classes = len(label_encoder.classes_)

# Split into features and labels
X = data.drop(columns=['Class'])
y = to_categorical(data['Class'], num_classes)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for deep learning models
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# CNN model
cnn_model = Sequential([
    Conv1D(64, 1, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# BiLSTM model
bilstm_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2), input_shape=(1, X.shape[2])),
    Bidirectional(LSTM(64, dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
bilstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
bilstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# GRU model
gru_model = Sequential([
    GRU(64, return_sequences=True, dropout=0.2, input_shape=(1, X.shape[2])),
    GRU(64, dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
gru_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
gru_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(f"\n{'='*40}\nEvaluation Results for {model_name}\n{'='*40}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")

# Run evaluation
evaluate_model(cnn_model, X_test, y_test, "CNN")
evaluate_model(bilstm_model, X_test, y_test, "BiLSTM")
evaluate_model(gru_model, X_test, y_test, "GRU")
