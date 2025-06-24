# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:25:13 2024

@author: cavus
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Dropout, BatchNormalization,
    concatenate, Attention, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

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
encoder = OneHotEncoder(sparse_output=False)
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

# Define the model (MFR-Net)
input_layer = Input(shape=(X.shape[1],))

# Dense layers for spatial and demographic fusion
fusion_layer = Dense(128, activation='relu')(input_layer)
fusion_layer = BatchNormalization()(fusion_layer)
fusion_layer = Dropout(0.3)(fusion_layer)

# Attention mechanism for critical feature selection
attention_input = Reshape((1, 128))(fusion_layer)  # Reshape for attention
attention_weights = Dense(128, activation='softmax')(fusion_layer)
attention_output = Attention()([attention_input, attention_input])
attention_output = Reshape((128,))(attention_output)

# Separate branches for perpetrator and victim predictions
perp_branch = Dense(64, activation='relu')(attention_output)
perp_branch = Dropout(0.3)(perp_branch)
perp_output = Dense(len(label_encoder_perp.classes_), activation='softmax', name='PERP_OUTPUT')(perp_branch)

vic_branch = Dense(64, activation='relu')(attention_output)
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
plt.title('MFR-Net Model Accuracy During Training', fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.savefig(r'C:\Users\cavus\Desktop\AI and Law Paper 2\MFR_Net_Training_Accuracy.png', dpi=600)
plt.show()




















