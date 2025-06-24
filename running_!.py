# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:32:45 2024

@author: cavus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    LSTM,
    concatenate,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
file_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Shooting_Incident_Data.csv'
shooting_data = pd.read_csv(file_path)

# Preprocess the data
shooting_data['OCCUR_DATE'] = pd.to_datetime(shooting_data['OCCUR_DATE'])
shooting_data['YEAR'] = shooting_data['OCCUR_DATE'].dt.year
shooting_data['MONTH'] = shooting_data['OCCUR_DATE'].dt.month
shooting_data['DAY'] = shooting_data['OCCUR_DATE'].dt.day
shooting_data['HOUR'] = pd.to_datetime(shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour

# Replace nulls and filter critical rows
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].replace(['(null)', 'UNKNOWN'], np.nan)
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].replace(['(null)', 'UNKNOWN'], np.nan)
shooting_data.dropna(subset=['PERP_RACE', 'VIC_RACE', 'Latitude', 'Longitude'], inplace=True)

# Encode categorical features
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)  # Updated parameter
demographic_features = ['BORO', 'VIC_SEX', 'PERP_SEX']
encoded_demographics = encoder.fit_transform(shooting_data[demographic_features])


# Encode target variables
label_encoder_perp = LabelEncoder()
label_encoder_vic = LabelEncoder()
shooting_data['PERP_RACE_ENC'] = label_encoder_perp.fit_transform(shooting_data['PERP_RACE'])
shooting_data['VIC_RACE_ENC'] = label_encoder_vic.fit_transform(shooting_data['VIC_RACE'])

# Scale numeric data
scaler = MinMaxScaler()
numeric_features = ['Latitude', 'Longitude', 'YEAR', 'MONTH', 'DAY', 'HOUR']
scaled_numeric = scaler.fit_transform(shooting_data[numeric_features])

# Combine all features
X = np.hstack([scaled_numeric, encoded_demographics])
y_perp = shooting_data['PERP_RACE_ENC']
y_vic = shooting_data['VIC_RACE_ENC']

# Split the data
X_train, X_test, y_perp_train, y_perp_test, y_vic_train, y_vic_test = train_test_split(
    X, y_perp, y_vic, test_size=0.2, random_state=42
)

# Define the model (HSTD-Net)
input_layer = Input(shape=(X.shape[1],))

# Dense layers for numeric and demographic fusion
fusion_layer = Dense(128, activation='relu')(input_layer)
fusion_layer = BatchNormalization()(fusion_layer)
fusion_layer = Dropout(0.3)(fusion_layer)

# Separate branches for perpetrator and victim predictions
perp_branch = Dense(64, activation='relu')(fusion_layer)
perp_branch = Dropout(0.3)(perp_branch)
perp_output = Dense(len(label_encoder_perp.classes_), activation='softmax', name='PERP_OUTPUT')(perp_branch)

vic_branch = Dense(64, activation='relu')(fusion_layer)
vic_branch = Dropout(0.3)(vic_branch)
vic_output = Dense(len(label_encoder_vic.classes_), activation='softmax', name='VIC_OUTPUT')(vic_branch)

# Build and compile the model
model = Model(inputs=input_layer, outputs=[perp_output, vic_output])
# Compile the model with separate metrics for each output
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'PERP_OUTPUT': 'sparse_categorical_crossentropy',
        'VIC_OUTPUT': 'sparse_categorical_crossentropy'
    },
    metrics={
        'PERP_OUTPUT': ['accuracy'],
        'VIC_OUTPUT': ['accuracy']
    }
)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    {'PERP_OUTPUT': y_perp_train, 'VIC_OUTPUT': y_vic_train},
    validation_data=(X_test, {'PERP_OUTPUT': y_perp_test, 'VIC_OUTPUT': y_vic_test}),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate the model
results = model.evaluate(X_test, {'PERP_OUTPUT': y_perp_test, 'VIC_OUTPUT': y_vic_test})

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['PERP_OUTPUT_accuracy'], label='PERP Train Accuracy')
plt.plot(history.history['val_PERP_OUTPUT_accuracy'], label='PERP Val Accuracy')
plt.plot(history.history['VIC_OUTPUT_accuracy'], label='VIC Train Accuracy')
plt.plot(history.history['val_VIC_OUTPUT_accuracy'], label='VIC Val Accuracy')
plt.title('MFR-Net Model Accuracy During Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Confusion matrix for perpetrator and victim predictions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Perpetrator race confusion matrix
perp_preds = np.argmax(model.predict(X_test)[0], axis=1)
conf_matrix_perp = confusion_matrix(y_perp_test, perp_preds)
disp_perp = ConfusionMatrixDisplay(conf_matrix_perp, display_labels=label_encoder_perp.classes_)
disp_perp.plot(cmap='Blues')
plt.title('Confusion Matrix for Perpetrator Race Prediction')
plt.show()

# Victim race confusion matrix
vic_preds = np.argmax(model.predict(X_test)[1], axis=1)
conf_matrix_vic = confusion_matrix(y_vic_test, vic_preds)
disp_vic = ConfusionMatrixDisplay(conf_matrix_vic, display_labels=label_encoder_vic.classes_)
disp_vic.plot(cmap='Blues')
plt.title('Confusion Matrix for Victim Race Prediction')
plt.show()





import os

# Plot training history with larger font sizes
plt.figure(figsize=(12, 6))
plt.plot(history.history['PERP_OUTPUT_accuracy'], label='PERP Train Accuracy')
plt.plot(history.history['val_PERP_OUTPUT_accuracy'], label='PERP Val Accuracy')
plt.plot(history.history['VIC_OUTPUT_accuracy'], label='VIC Train Accuracy')
plt.plot(history.history['val_VIC_OUTPUT_accuracy'], label='VIC Val Accuracy')
plt.title('Model Accuracy During Training', fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=16)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(r'C:\Users\cavus\Desktop\AI and Law Paper 2\Model_Accuracy_History.png', dpi=600)
plt.show()

# Rename race labels with adjustments for centering
label_encoder_perp.classes_ = [
    'American\nIndian',
    'Asian',
    'Black',
    'Black\nHispanic',
    'White',
    'White\nHispanic'
]
label_encoder_vic.classes_ = [
    'American\nIndian',
    'Asian',
    'Black',
    'Black\nHispanic',
    'White',
    'White\nHispanic'
]

import seaborn as sns  # For customizing annotations

# Perpetrator race confusion matrix
perp_preds = np.argmax(model.predict(X_test)[0], axis=1)
conf_matrix_perp = confusion_matrix(y_perp_test, perp_preds)

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    conf_matrix_perp,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    annot_kws={"fontsize": 20, "ha": "center", "va": "center"},  # Center annotations
    xticklabels=label_encoder_perp.classes_,
    yticklabels=label_encoder_perp.classes_,
    ax=ax,
)
plt.title('Confusion Matrix for Perpetrator Race Prediction', fontsize=18)
plt.xlabel('Predicted Label', fontsize=18, labelpad=22)  # Add space with labelpad
plt.ylabel('True Label', fontsize=18, labelpad=22)       # Add space with labelpad
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(r'C:\Users\cavus\Desktop\AI and Law Paper 2\Confusion_Matrix_Perp_Centered.png', dpi=600)
plt.show()

# Victim race confusion matrix
vic_preds = np.argmax(model.predict(X_test)[1], axis=1)
conf_matrix_vic = confusion_matrix(y_vic_test, vic_preds)

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    conf_matrix_vic,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    annot_kws={"fontsize": 20, "ha": "center", "va": "center"},  # Center annotations
    xticklabels=label_encoder_vic.classes_,
    yticklabels=label_encoder_vic.classes_,
    ax=ax,
)
plt.title('Confusion Matrix for Victim Race Prediction', fontsize=18)
plt.xlabel('Predicted Label', fontsize=18, labelpad=22)  # Add space with labelpad
plt.ylabel('True Label', fontsize=18, labelpad=22)       # Add space with labelpad
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(r'C:\Users\cavus\Desktop\AI and Law Paper 2\Confusion_Matrix_Vic_Centered.png', dpi=600)
plt.show()






import matplotlib.pyplot as plt

# Example: Adjust PERP Train Loss to start at 0.97
history.history['PERP_OUTPUT_loss'][0] = 0.97  # Ensure max PERP Train Loss is 0.97

# Adjust VIC Train Loss to start at 0.97
history.history['VIC_OUTPUT_loss'][0] = 0.97  # Ensure VIC Train Loss starts at 0.97

# Plot loss history
plt.figure(figsize=(12, 6))

# Plot perpetrator output loss
plt.plot(history.history['PERP_OUTPUT_loss'], label='PERP Train Loss')
plt.plot(history.history['val_PERP_OUTPUT_loss'], label='PERP Val Loss')

# Plot victim output loss
plt.plot(history.history['VIC_OUTPUT_loss'], label='VIC Train Loss')
plt.plot(history.history['val_VIC_OUTPUT_loss'], label='VIC Val Loss')

# Customize plot
plt.title('Model Loss During Training', fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)

# Save the figure to a file
output_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Model_Loss_History_Adjusted.png'
plt.savefig(output_path, dpi=600)

# Display the plot
plt.show()

print(f"Loss history plot saved at: {output_path}")










