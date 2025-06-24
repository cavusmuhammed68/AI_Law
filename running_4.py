# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:01:29 2024

@author: cavus
"""

# 1. Data Preparation

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\AI and Law Paper 2\Shooting_Incident_Data.csv'
shooting_data = pd.read_csv(file_path)

# Normalize race columns
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].str.upper()
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].str.upper()

# Replace '(null)' and 'UNKNOWN' with NaN
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].replace(['(NULL)', 'UNKNOWN'], np.nan)
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].replace(['(NULL)', 'UNKNOWN'], np.nan)

# Drop rows with NaN in critical columns
shooting_data.dropna(subset=['PERP_RACE', 'VIC_RACE', 'Latitude', 'Longitude'], inplace=True)

# Feature engineering
shooting_data['YEAR'] = pd.to_datetime(shooting_data['OCCUR_DATE']).dt.year
shooting_data['MONTH'] = pd.to_datetime(shooting_data['OCCUR_DATE']).dt.month
shooting_data['DAY'] = pd.to_datetime(shooting_data['OCCUR_DATE']).dt.day
shooting_data['HOUR'] = pd.to_datetime(shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour


# 2. Feature Encoding and Data Splitting

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
demographic_features = ['BORO', 'VIC_SEX', 'PERP_SEX']
encoded_demographics = encoder.fit_transform(shooting_data[demographic_features])

# Encode target variables
label_encoder_perp = LabelEncoder()
shooting_data['PERP_RACE_ENC'] = label_encoder_perp.fit_transform(shooting_data['PERP_RACE'])

label_encoder_vic = LabelEncoder()
shooting_data['VIC_RACE_ENC'] = label_encoder_vic.fit_transform(shooting_data['VIC_RACE'])

# Scale numeric data
scaler = MinMaxScaler()
numeric_features = ['Latitude', 'Longitude', 'YEAR', 'MONTH', 'DAY', 'HOUR']
scaled_numeric = scaler.fit_transform(shooting_data[numeric_features])

# Combine all features
X = np.hstack([scaled_numeric, encoded_demographics])
y_perp = shooting_data['PERP_RACE_ENC']
y_vic = shooting_data['VIC_RACE_ENC']
protected_attribute = shooting_data['PERP_RACE_ENC']

# Split the data
X_train, X_test, y_perp_train, y_perp_test, y_vic_train, y_vic_test, prot_train, prot_test = train_test_split(
    X, y_perp, y_vic, protected_attribute, test_size=0.2, random_state=42
)


# 3. Model Design and Training

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GRU, MultiHeadAttention, Flatten, Reshape, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the model
input_layer = Input(shape=(X.shape[1],))
reshaped_input = Reshape((X.shape[1], 1))(input_layer)

# GRU Layer
gru_layer = GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(reshaped_input)

# Multi-Head Attention Layer
attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(gru_layer, gru_layer)
attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

# Flatten and Dense Layers
attention_flattened = Flatten()(attention_output)
fusion_layer = Dense(256, activation='relu')(attention_flattened)
fusion_layer = BatchNormalization()(fusion_layer)
fusion_layer = Dropout(0.4)(fusion_layer)

# Branches for perpetrator, victim, and adversarial tasks
perp_output = Dense(len(label_encoder_perp.classes_), activation='softmax', name='PERP_OUTPUT')(Dense(128, activation='relu')(fusion_layer))
vic_output = Dense(len(label_encoder_vic.classes_), activation='softmax', name='VIC_OUTPUT')(Dense(128, activation='relu')(fusion_layer))
adversary_output = Dense(len(np.unique(protected_attribute)), activation='softmax', name='ADVERSARY_OUTPUT')(Dense(128, activation='relu')(fusion_layer))

# Build and Compile the Model
model = Model(inputs=input_layer, outputs=[perp_output, vic_output, adversary_output])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'PERP_OUTPUT': 'sparse_categorical_crossentropy', 'VIC_OUTPUT': 'sparse_categorical_crossentropy', 'ADVERSARY_OUTPUT': 'sparse_categorical_crossentropy'},
    loss_weights={'PERP_OUTPUT': 1.0, 'VIC_OUTPUT': 1.0, 'ADVERSARY_OUTPUT': 0.5},
    metrics={'PERP_OUTPUT': ['accuracy'], 'VIC_OUTPUT': ['accuracy'], 'ADVERSARY_OUTPUT': ['accuracy']}
)

# Train the Model
history = model.fit(
    X_train,
    {'PERP_OUTPUT': y_perp_train, 'VIC_OUTPUT': y_vic_train, 'ADVERSARY_OUTPUT': prot_train},
    validation_data=(X_test, {'PERP_OUTPUT': y_perp_test, 'VIC_OUTPUT': y_vic_test, 'ADVERSARY_OUTPUT': prot_test}),
    epochs=100,
    batch_size=64,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]
)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Plot training accuracy
def plot_training_accuracy(history, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['PERP_OUTPUT_accuracy'], label='PERP Train Accuracy')
    plt.plot(history.history['val_PERP_OUTPUT_accuracy'], label='PERP Val Accuracy')
    plt.plot(history.history['VIC_OUTPUT_accuracy'], label='VIC Train Accuracy')
    plt.plot(history.history['val_VIC_OUTPUT_accuracy'], label='VIC Val Accuracy')
    plt.title('Training Accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

plot_training_accuracy(history, r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Training_Accuracy.png')

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Generate confusion matrices
y_perp_pred = np.argmax(model.predict(X_test)[0], axis=1)
y_vic_pred = np.argmax(model.predict(X_test)[1], axis=1)
updated_labels = ['American\nIndian', 'Asian', 'Black', 'Black\nHispanic', 'White', 'White\nHispanic']

plot_confusion_matrix(y_perp_test, y_perp_pred, updated_labels, 'Perpetrator Confusion Matrix', r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Confusion_Matrix_Perpetrator.png')
plot_confusion_matrix(y_vic_test, y_vic_pred, updated_labels, 'Victim Confusion Matrix', r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Confusion_Matrix_Victim.png')




# 5. Fairness Metrics and Crime Map

import folium
import numpy as np

# Prepare data for visualization
real_data_locations = list(zip(shooting_data['Latitude'], shooting_data['Longitude']))
crime_likelihood = model.predict(X)[0].max(axis=1)  # Assuming max likelihood for simplicity

# Assign colors based on crime likelihood
crime_colors = [
    'green' if likelihood < 0.65 else 'red'  # Threshold adjusted to 0.6
    for likelihood in crime_likelihood
]

# Create a Folium map centered on the average location
crime_map = folium.Map(
    location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()],
    zoom_start=12
)

# Add circle markers for each data point
for i, (lat, lon) in enumerate(real_data_locations):
    folium.CircleMarker(
        location=(lat, lon),
        radius=0.5,  # Adjusted marker size for better visibility
        color=crime_colors[i],
        fill=True,
        fill_opacity=0.7,
        popup=f"Crime Likelihood: {crime_likelihood[i]:.2f}"
    ).add_to(crime_map)

# Corrected draggable legend
legend_html = """
<div id="legend" style="position: absolute; 
                         top: 50px; right: 50px; 
                         width: 250px; height: auto; 
                         border:2px solid grey; 
                         z-index:9999; 
                         font-size:14px; 
                         background-color:white; 
                         padding: 10px; 
                         border-radius: 8px; 
                         box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
<b>Crime Possibility </b><br>
&emsp;<span style="color:green;">&#9679;</span> Low Possible  <br>
&emsp;<span style="color:red;">&#9679;</span> High Possible  <br>
</div>

<script>
    const legend = document.getElementById('legend');
    let isMouseDown = false, offsetX, offsetY;

    legend.addEventListener('mousedown', function(e) {
        isMouseDown = true;
        offsetX = e.clientX - legend.offsetLeft;
        offsetY = e.clientY - legend.offsetTop;
        legend.style.cursor = 'grabbing';
    });

    document.addEventListener('mouseup', function() {
        isMouseDown = false;
        legend.style.cursor = 'grab';
    });

    document.addEventListener('mousemove', function(e) {
        if (isMouseDown) {
            const newLeft = e.clientX - offsetX;
            const newTop = e.clientY - offsetY;
            legend.style.left = `${newLeft}px`;
            legend.style.top = `${newTop}px`;
        }
    });
</script>
"""

# Add the legend to the map
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
map_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Crime_Map_Draggable_Legend.html'
crime_map.save(map_path)

print(f"Crime map saved to: {map_path}")





# Get predictions for perpetrators and victims
y_perp_pred_post = np.argmax(model.predict(X_test)[0], axis=1)  # Perpetrator predictions
y_vic_pred_post = np.argmax(model.predict(X_test)[1], axis=1)   # Victim predictions

# Fairness metrics calculation
def calculate_fairness_metrics(y_true, y_pred, protected_attr, group_names):
    metrics = {}
    for i, group in enumerate(group_names):
        group_mask = protected_attr == i  # Mask for the current group
        if group_mask.sum() == 0:
            continue  # Skip groups with no samples
        true_group = y_true[group_mask]
        pred_group = y_pred[group_mask]

        # Calculate metrics
        tp = np.sum((pred_group == i) & (true_group == i))
        fp = np.sum((pred_group == i) & (true_group != i))
        fn = np.sum((pred_group != i) & (true_group == i))
        tn = np.sum((pred_group != i) & (true_group != i))

        tpr = tp / (tp + fn) if tp + fn > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if fp + tn > 0 else 0  # False Positive Rate

        metrics[group] = {'TPR': tpr, 'FPR': fpr}
    return metrics

# Apply metrics
group_names = label_encoder_perp.classes_  # Use perpetrator classes as an example
perp_fairness_metrics_post = calculate_fairness_metrics(y_perp_test, y_perp_pred_post, prot_test, group_names)
vic_fairness_metrics_post = calculate_fairness_metrics(y_vic_test, y_vic_pred_post, prot_test, group_names)

print("Post-Mitigation Perpetrator Fairness Metrics:")
for group, metric in perp_fairness_metrics_post.items():
    print(f"{group}: {metric}")

print("\nPost-Mitigation Victim Fairness Metrics:")
for group, metric in vic_fairness_metrics_post.items():
    print(f"{group}: {metric}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"fontsize": 16}
    )
    plt.title(title, fontsize=18)
    plt.xlabel('Predicted Labels', fontsize=16, labelpad=20)
    plt.ylabel('True Labels', fontsize=16, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Updated labels for races
updated_labels = [
    'American\nIndian', 'Asian', 'Black', 
    'Black\nHispanic', 'White', 'White\nHispanic'
]

# Perpetrator Confusion Matrix (Post-Mitigation)
plot_confusion_matrix(
    y_perp_test, y_perp_pred_post, updated_labels,
    'Confusion Matrix for Perpetrator Prediction (Post-Mitigation)',
    r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Confusion_Matrix_Perp_Post_Mitigation.png'
)

# Victim Confusion Matrix (Post-Mitigation)
plot_confusion_matrix(
    y_vic_test, y_vic_pred_post, updated_labels,
    'Confusion Matrix for Victim Prediction (Post-Mitigation)',
    r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Confusion_Matrix_Vic_Post_Mitigation.png'
)

# Function to visualize fairness metrics
def plot_fairness_metrics(metrics, title, filename):
    groups = list(metrics.keys())
    tpr = [metrics[group]['TPR'] for group in groups]
    fpr = [metrics[group]['FPR'] for group in groups]

    x = np.arange(len(groups))  # Group positions
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, tpr, width, label='True Positive Rate')
    bars2 = ax.bar(x + width/2, fpr, width, label='False Positive Rate')

    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel('Rate', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add bar values
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Plot fairness metrics
plot_fairness_metrics(
    perp_fairness_metrics_post,
    'Post-Mitigation Fairness Metrics for Perpetrator Prediction',
    r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Fairness_Metrics_Perp_Post.png'
)
plot_fairness_metrics(
    vic_fairness_metrics_post,
    'Post-Mitigation Fairness Metrics for Victim Prediction',
    r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Fairness_Metrics_Vic_Post.png'
)


import folium
import numpy as np

# Define thresholds for guilt classification
def classify_guilt(likelihood):
    if likelihood < 0.65:  # Correct threshold per user request
        return 'Low Possible Guilt'
    else:
        return 'High Possible Guilt'

# Prepare the map data
real_data_locations = list(zip(shooting_data['Latitude'], shooting_data['Longitude']))
crime_likelihood = model.predict(X)[0].max(axis=1)  # Use the maximum likelihood for simplicity

# Assign colors based on guilt category
crime_colors = [
    'green' if classify_guilt(likelihood) == 'Low Possible Guilt' else 'red' 
    for likelihood in crime_likelihood
]

# Create a Folium map centered on the average location
crime_map = folium.Map(
    location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()],
    zoom_start=12
)

# Add circle markers for each data point
for i, (lat, lon) in enumerate(real_data_locations):
    guilt_status = classify_guilt(crime_likelihood[i])
    folium.CircleMarker(
        location=(lat, lon),
        radius=0.5,  # Adjusted marker size for better visibility
        color=crime_colors[i],
        fill=True,
        fill_opacity=0.7,
        popup=f"Crime Likelihood: {crime_likelihood[i]:.2f}<br>Guilt Category: {guilt_status}"
    ).add_to(crime_map)

# Corrected draggable legend
legend_html = """
<div id="legend" style="position: absolute; 
                         top: 50px; right: 50px; 
                         width: 250px; height: auto; 
                         border:2px solid grey; 
                         z-index:9999; 
                         font-size:18px; 
                         background-color:white; 
                         padding: 10px; 
                         border-radius: 8px; 
                         box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
<b>Crime Possibility </b><br>
&emsp;<span style="color:green;">&#9679;</span> Low Possible  <br>
&emsp;<span style="color:red;">&#9679;</span> High Possible  <br>
</div>

<script>
    const legend = document.getElementById('legend');
    let isMouseDown = false, offsetX, offsetY;

    legend.addEventListener('mousedown', function(e) {
        isMouseDown = true;
        offsetX = e.clientX - legend.offsetLeft;
        offsetY = e.clientY - legend.offsetTop;
        legend.style.cursor = 'grabbing';
    });

    document.addEventListener('mouseup', function() {
        isMouseDown = false;
        legend.style.cursor = 'grab';
    });

    document.addEventListener('mousemove', function(e) {
        if (isMouseDown) {
            const newLeft = e.clientX - offsetX;
            const newTop = e.clientY - offsetY;
            legend.style.left = `${newLeft}px`;
            legend.style.top = `${newTop}px`;
        }
    });
</script>
"""

# Add the legend to the map
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
map_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Real_Time_Guilt_Map.html'
crime_map.save(map_path)

print(f"Real-time crime map with guilt categories saved to: {map_path}")





import folium
from sklearn.ensemble import IsolationForest
import numpy as np

# Define thresholds for guilt classification
def classify_guilt(likelihood, anomaly):
    if anomaly == -1:
        return 'Anomalous'
    elif likelihood < 0.65:
        return 'Low Possible Guilt'
    else:
        return 'High Possible Guilt'

# Prepare the map data
real_data_locations = list(zip(shooting_data['Latitude'], shooting_data['Longitude']))
crime_likelihood = model.predict(X)[0].max(axis=1)  # Use the maximum likelihood for simplicity

# Use Isolation Forest for anomaly detection
isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # Set contamination to 5%
anomaly_labels = isolation_forest.fit_predict(crime_likelihood.reshape(-1, 1))  # Detect anomalies (-1 = anomaly)

# Assign colors based on guilt category and anomaly detection
crime_colors = []
for i, likelihood in enumerate(crime_likelihood):
    if anomaly_labels[i] == -1:  # Anomalous point
        crime_colors.append('purple')  # Purple for anomalies
    else:
        crime_colors.append('green' if likelihood < 0.65 else 'red')  # Green or red for normal points

# Create a Folium map centered on the average location
crime_map = folium.Map(
    location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()],
    zoom_start=12
)

# Add circle markers for each data point
for i, (lat, lon) in enumerate(real_data_locations):
    guilt_status = classify_guilt(crime_likelihood[i], anomaly_labels[i])
    folium.CircleMarker(
        location=(lat, lon),
        radius=0.5 if guilt_status == 'Anomalous' else 0.5,  # Larger radius for anomalies
        color=crime_colors[i],
        fill=True,
        fill_opacity=0.7,
        popup=f"Crime Likelihood: {crime_likelihood[i]:.2f}<br>Guilt Category: {guilt_status}"
    ).add_to(crime_map)

# Add corrected draggable legend
legend_html = """
<div id="legend" style="position: absolute; 
                         top: 50px; right: 50px; 
                         width: 250px; height: auto; 
                         border:2px solid grey; 
                         z-index:9999; 
                         font-size:18px; 
                         background-color:white; 
                         padding: 10px; 
                         border-radius: 8px; 
                         box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
<b>Crime Possibility</b><br>
&emsp;<span style="color:green;">&#9679;</span> Low Possible  <br>
&emsp;<span style="color:red;">&#9679;</span> High Possible <br>
&emsp;<span style="color:purple;">&#9679;</span> Anomalous Data Point<br>
</div>

<script>
    const legend = document.getElementById('legend');
    let isMouseDown = false, offsetX, offsetY;

    legend.addEventListener('mousedown', function(e) {
        isMouseDown = true;
        offsetX = e.clientX - legend.offsetLeft;
        offsetY = e.clientY - legend.offsetTop;
        legend.style.cursor = 'grabbing';
    });

    document.addEventListener('mouseup', function() {
        isMouseDown = false;
        legend.style.cursor = 'grab';
    });

    document.addEventListener('mousemove', function(e) {
        if (isMouseDown) {
            legend.style.left = `${e.clientX - offsetX}px`;
            legend.style.top = `${e.clientY - offsetY}px`;
        }
    });
</script>
"""

# Add the legend to the map
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
map_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\Crime_Map_With_Anomaly_Detection_Fixed.html'
crime_map.save(map_path)

print(f"Crime map with anomaly detection saved to: {map_path}")




from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam



def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²:  {r2:.4f}\n")
    
    return mse, mae, r2


# For fair comparison — use PERP_RACE_ENC as numeric target
y_target_train = y_perp_train
y_target_test = y_perp_test


# Get predicted class probabilities → convert to argmax (integer predictions)
y_pred_mftnet = np.argmax(model.predict(X_test)[0], axis=1)

# Evaluate
mse_mft, mae_mft, r2_mft = evaluate_model(y_target_test, y_pred_mftnet, "MFT-Net (Proposed)")


# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_target_train)
y_pred_rf = rf.predict(X_test)

# Evaluate
mse_rf, mae_rf, r2_rf = evaluate_model(y_target_test, y_pred_rf, "Random Forest")


# Reshape X for LSTM (samples, timesteps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1]), dropout=0.2, recurrent_dropout=0.2),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train
lstm_model.fit(X_train_lstm, y_target_train, epochs=50, batch_size=64, verbose=1, validation_data=(X_test_lstm, y_target_test))

# Predict
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

# Evaluate
mse_lstm, mae_lstm, r2_lstm = evaluate_model(y_target_test, y_pred_lstm, "LSTM")


# Reshape for CNN (samples, timesteps, features)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define CNN model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(1)
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train
cnn_model.fit(X_train_cnn, y_target_train, epochs=50, batch_size=64, verbose=1, validation_data=(X_test_cnn, y_target_test))

# Predict
y_pred_cnn = cnn_model.predict(X_test_cnn).flatten()

# Evaluate
mse_cnn, mae_cnn, r2_cnn = evaluate_model(y_target_test, y_pred_cnn, "CNN")


# Print all in table format
print("\nFinal Summary (For Paper Table):")
print(f"{'Model':<25} {'MSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 50)
print(f"{'MFT-Net (Proposed)':<25} {mse_mft:.4f}    {mae_mft:.4f}    {r2_mft:.4f}")
print(f"{'Random Forest':<25} {mse_rf:.4f}    {mae_rf:.4f}    {r2_rf:.4f}")
print(f"{'LSTM':<25} {mse_lstm:.4f}    {mae_lstm:.4f}    {r2_lstm:.4f}")
print(f"{'CNN':<25} {mse_cnn:.4f}    {mae_cnn:.4f}    {r2_cnn:.4f}")
















