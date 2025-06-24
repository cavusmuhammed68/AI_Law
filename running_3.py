import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Shooting_Incident_Data.csv'
shooting_data = pd.read_csv(file_path)

# Normalize race columns to ensure consistent formatting
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].str.upper()
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].str.upper()

# Replace '(null)' and 'UNKNOWN' with NaN
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].replace(['(NULL)', 'UNKNOWN'], np.nan)
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].replace(['(NULL)', 'UNKNOWN'], np.nan)

# Drop rows with NaN in critical columns
shooting_data.dropna(subset=['PERP_RACE', 'VIC_RACE', 'Latitude', 'Longitude'], inplace=True)

# Feature engineering for time-based data
shooting_data['YEAR'] = pd.to_datetime(shooting_data['OCCUR_DATE']).dt.year
shooting_data['MONTH'] = pd.to_datetime(shooting_data['OCCUR_DATE']).dt.month
shooting_data['DAY'] = pd.to_datetime(shooting_data['OCCUR_DATE']).dt.day
shooting_data['HOUR'] = pd.to_datetime(shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour

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
protected_attribute = shooting_data['PERP_RACE_ENC']  # Protected attribute for fairness

# Split the data into training and testing sets
X_train, X_test, y_perp_train, y_perp_test, y_vic_train, y_vic_test, prot_train, prot_test = train_test_split(
    X, y_perp, y_vic, protected_attribute, test_size=0.2, random_state=42
)


from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GRU, MultiHeadAttention, Flatten, Reshape, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the improved MFR-Net model
input_layer = Input(shape=(X.shape[1],))

# Reshape input for GRU compatibility
reshaped_input = Reshape((X.shape[1], 1))(input_layer)

# GRU Layer for feature extraction
gru_layer = GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(reshaped_input)

# Multi-Head Attention Layer with Layer Normalization
attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(gru_layer, gru_layer)
attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

# Flatten attention output and add dense layers
attention_flattened = Flatten()(attention_output)
fusion_layer = Dense(256, activation='relu')(attention_flattened)
fusion_layer = BatchNormalization()(fusion_layer)
fusion_layer = Dropout(0.4)(fusion_layer)

# Separate branches for perpetrator and victim predictions
perp_branch = Dense(128, activation='relu')(fusion_layer)
perp_branch = Dropout(0.3)(perp_branch)
perp_output = Dense(len(label_encoder_perp.classes_), activation='softmax', name='PERP_OUTPUT')(perp_branch)

vic_branch = Dense(128, activation='relu')(fusion_layer)
vic_branch = Dropout(0.3)(vic_branch)
vic_output = Dense(len(label_encoder_vic.classes_), activation='softmax', name='VIC_OUTPUT')(vic_branch)

# Adversarial Network for Bias Mitigation
adversary_branch = Dense(128, activation='relu')(fusion_layer)
adversary_branch = Dropout(0.3)(adversary_branch)
adversary_output = Dense(len(np.unique(protected_attribute)), activation='softmax', name='ADVERSARY_OUTPUT')(adversary_branch)

# Build the combined model
model = Model(inputs=input_layer, outputs=[perp_output, vic_output, adversary_output])

# Compile the model with weighted loss for imbalance handling
loss_weights = {'PERP_OUTPUT': 1.0, 'VIC_OUTPUT': 1.0, 'ADVERSARY_OUTPUT': 0.5}  # Reduce weight for adversary
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'PERP_OUTPUT': 'sparse_categorical_crossentropy',
        'VIC_OUTPUT': 'sparse_categorical_crossentropy',
        'ADVERSARY_OUTPUT': 'sparse_categorical_crossentropy'
    },
    loss_weights=loss_weights,
    metrics={
        'PERP_OUTPUT': ['accuracy'],
        'VIC_OUTPUT': ['accuracy'],
        'ADVERSARY_OUTPUT': ['accuracy']
    }
)

# Train the improved model
history = model.fit(
    X_train,
    {
        'PERP_OUTPUT': y_perp_train,
        'VIC_OUTPUT': y_vic_train,
        'ADVERSARY_OUTPUT': prot_train
    },
    validation_data=(X_test, {
        'PERP_OUTPUT': y_perp_test,
        'VIC_OUTPUT': y_vic_test,
        'ADVERSARY_OUTPUT': prot_test
    }),
    epochs=100,  # Train for more epochs
    batch_size=64,  # Increase batch size
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]
)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define a corrected mapping for race labels
race_label_mapping = {
    "AMERICAN INDIAN/ALASKAN NATIVE": "American Indian",
    "ASIAN / PACIFIC ISLANDER": "Asian",  # Correctly handle the extra space
    "BLACK": "Black",
    "BLACK HISPANIC": "Black Hispanic",
    "WHITE": "White",
    "WHITE HISPANIC": "White Hispanic"
}

# Update labels for perpetrator and victim races
updated_perp_labels = [race_label_mapping.get(race, race) for race in label_encoder_perp.classes_]
updated_vic_labels = [race_label_mapping.get(race, race) for race in label_encoder_vic.classes_]

# Evaluate the model
results = model.evaluate(X_test, {
    'PERP_OUTPUT': y_perp_test,
    'VIC_OUTPUT': y_vic_test,
    'ADVERSARY_OUTPUT': prot_test
})

# Predictions
y_perp_pred = np.argmax(model.predict(X_test)[0], axis=1)
y_vic_pred = np.argmax(model.predict(X_test)[1], axis=1)

# Function to plot training accuracy
def plot_training_accuracy(history, filename):
    plt.figure(figsize=(12, 6))
    
    # PERP_OUTPUT accuracy
    plt.plot(history.history['PERP_OUTPUT_accuracy'], label='PERP Train Accuracy', color='blue')
    plt.plot(history.history['val_PERP_OUTPUT_accuracy'], label='PERP Val Accuracy', color='orange')
    
    # VIC_OUTPUT accuracy
    plt.plot(history.history['VIC_OUTPUT_accuracy'], label='VIC Train Accuracy', color='green')
    plt.plot(history.history['val_VIC_OUTPUT_accuracy'], label='VIC Val Accuracy', color='red')
    
    # Chart details
    plt.title('MFR-Net Model Accuracy During Training', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=600)
    plt.show()

# Generate the training accuracy plot
plot_training_accuracy(
    history,
    r'C:\Users\cavus\Desktop\AI and Law Paper 2\MFR_Net_Training_Accuracy.png'
)


# Corrected fairness metrics calculation
def fairness_metrics(y_true, y_pred, prot_attr):
    # Create a DataFrame for analysis
    df = pd.DataFrame({'True': y_true, 'Predicted': y_pred, 'Protected': prot_attr})
    
    # Ensure protected attribute values are properly encoded
    privileged_group = df['Protected'] != label_encoder_perp.transform(['BLACK'])[0]
    unprivileged_group = df['Protected'] == label_encoder_perp.transform(['BLACK'])[0]
    
    # Handle cases with zero samples in either group
    if privileged_group.sum() == 0 or unprivileged_group.sum() == 0:
        raise ValueError("One of the groups (privileged/unprivileged) has zero samples.")
    
    privileged = df[privileged_group]
    unprivileged = df[unprivileged_group]

    # Calculate demographic parity rates
    privileged_rate = (privileged['Predicted'] == privileged['True']).mean() if len(privileged) > 0 else 0
    unprivileged_rate = (unprivileged['Predicted'] == unprivileged['True']).mean() if len(unprivileged) > 0 else 0

    # Calculate disparate impact
    disparate_impact = unprivileged_rate / privileged_rate if privileged_rate > 0 else 0

    # Calculate equal opportunity difference
    tpr_privileged = (privileged['True'] == privileged['Predicted']).mean() if len(privileged) > 0 else 0
    tpr_unprivileged = (unprivileged['True'] == unprivileged['Predicted']).mean() if len(unprivileged) > 0 else 0
    equal_opportunity_diff = tpr_unprivileged - tpr_privileged

    return {
        'Demographic Parity': (privileged_rate, unprivileged_rate),
        'Disparate Impact': disparate_impact,
        'Equal Opportunity Difference': equal_opportunity_diff
    }


# Calculate fairness metrics for victim and perpetrator predictions
victim_fairness = fairness_metrics(y_vic_test, y_vic_pred, prot_test)
perp_fairness = fairness_metrics(y_perp_test, y_perp_pred, prot_test)

# Recalculate and plot fairness metrics
victim_fairness = fairness_metrics(y_vic_test, y_vic_pred, prot_test)
perp_fairness = fairness_metrics(y_perp_test, y_perp_pred, prot_test)

# Function to visualize fairness metrics
def visualize_fairness_metrics(fairness_results, title, filename):
    privileged_rate, unprivileged_rate = fairness_results['Demographic Parity']
    disparate_impact = fairness_results['Disparate Impact']
    equal_opportunity_diff = fairness_results['Equal Opportunity Difference']

    metrics = ['Demographic Parity (Privileged)', 'Demographic Parity (Unprivileged)', 
               'Disparate Impact', 'Equal Opportunity Difference']
    values = [privileged_rate, unprivileged_rate, disparate_impact, equal_opportunity_diff]

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'purple', 'green']
    plt.bar(metrics, values, color=colors)
    plt.title(title, fontsize=16)
    plt.ylabel('Metric Value', fontsize=14)
    plt.ylim(0, 1.2)  # Adjust y-axis for readability
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(values):  # Annotate bars with values
        plt.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Plot fairness metrics for Victim predictions
visualize_fairness_metrics(
    victim_fairness,
    'Fairness Metrics for Victim Prediction (Post-Mitigation)',
    r'C:\Users\cavus\Desktop\AI and Law Paper 2\Fairness_Victim_Prediction_Corrected.png'
)

# Plot fairness metrics for Perpetrator predictions
visualize_fairness_metrics(
    perp_fairness,
    'Fairness Metrics for Perpetrator Prediction (Post-Mitigation)',
    r'C:\Users\cavus\Desktop\AI and Law Paper 2\Fairness_Perpetrator_Prediction_Corrected.png'
)


# Visualize fairness metrics for victim predictions
visualize_fairness_metrics(
    victim_fairness,
    'Fairness Metrics for Victim Prediction (Post-Mitigation)',
    r'C:\Users\cavus\Desktop\AI and Law Paper 2\Fairness_Victim_Prediction.png'
)

# Visualize fairness metrics for perpetrator predictions
visualize_fairness_metrics(
    perp_fairness,
    'Fairness Metrics for Perpetrator Prediction (Post-Mitigation)',
    r'C:\Users\cavus\Desktop\AI and Law Paper 2\Fairness_Perpetrator_Prediction.png'
)




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to plot confusion matrix with customized labels and larger numbers
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(12, 12))  # Larger size for clarity
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"fontsize": 20},  # Increase font size of numbers
        ax=ax,
    )
    
    # Customize title and labels
    plt.title(title, fontsize=18)
    plt.xlabel('Predicted Labels', fontsize=18, labelpad=22)  # Add label padding
    plt.ylabel('True Labels', fontsize=18, labelpad=22)       # Add label padding
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Updated labels for victim and perpetrator groups
updated_vic_labels = [
    'American\nIndian',
    'Asian',
    'Black',
    'Black\nHispanic',
    'White',
    'White\nHispanic'
]
updated_perp_labels = updated_vic_labels  # Assuming same labels for perpetrators

# Plot confusion matrix for victim prediction
plot_confusion_matrix(
    y_vic_test, y_vic_pred, updated_vic_labels,
    'Confusion Matrix for Victim Prediction (Post-Mitigation)',
    r'C:\Users\cavus\Desktop\AI and Law Paper 2\Confusion_Matrix_Victim_Post.png'
)

# Plot confusion matrix for perpetrator prediction
plot_confusion_matrix(
    y_perp_test, y_perp_pred, updated_perp_labels,
    'Confusion Matrix for Perpetrator Prediction (Post-Mitigation)',
    r'C:\Users\cavus\Desktop\AI and Law Paper 2\Confusion_Matrix_Perpetrator_Post.png'
)


import folium
import numpy as np

# Use real data points from the shooting_data dataset
real_data_inputs = []
real_data_locations = []

# Extract real locations and features for prediction
for index, row in shooting_data.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']
    
    # Scale input features using the same scaler as training
    scaled_features = scaler.transform([[latitude, longitude, row['YEAR'], row['MONTH'], row['DAY'], row['HOUR']]])
    
    # Combine with demographic features (use encoded demographic features)
    demographic_features = np.zeros(encoded_demographics.shape[1])  # Replace with real demographics if available
    full_features = np.hstack([scaled_features[0], demographic_features])
    
    real_data_inputs.append(full_features)
    real_data_locations.append((latitude, longitude))

# Convert inputs to NumPy array
real_data_inputs = np.array(real_data_inputs)

# Predict crime probabilities for real data points
predictions = model.predict(real_data_inputs)[0]  # Get perpetrator likelihood
crime_likelihood = np.max(predictions, axis=1).astype(float)  # Use maximum likelihood as crime possibility

# Define crime categories
crime_colors = ['green' if likelihood < 0.51 else 'red' for likelihood in crime_likelihood]

# Create a Folium map centered on the dataset's average location
crime_map = folium.Map(location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()], zoom_start=12)

# Add circle markers for each data point based on crime prediction results
for i, (lat, lon) in enumerate(real_data_locations):
    color = crime_colors[i]
    folium.CircleMarker(
        location=(lat, lon),  # Ensure lat and lon are real data points
        radius=4,  # Adjusted marker size for better visibility
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"Crime Likelihood: {crime_likelihood[i]:.2f}"
    ).add_to(crime_map)

# Add a fully draggable legend
legend_html = """
<div id='legend' style="position: absolute; 
            bottom: 50px; left: 50px; width: 250px; height: auto; 
            background-color: white; z-index: 1000; font-size: 14px; border:2px solid grey; padding: 10px; border-radius: 8px; cursor: grab;">
<b>Real-Time Crime Possibility</b><br>
<span style="color:green;">&#9679;</span> Low Crime Possibility <br>
<span style="color:red;">&#9679;</span> High Crime Possibility 
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
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
map_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Prediction_Based_Crime_Map_With_Fully_Draggable_Legend.html'
crime_map.save(map_path)

print(f"Prediction-based crime map with fully draggable legend saved to: {map_path}")





# with Anomaly Detection #

import folium
from sklearn.ensemble import IsolationForest
import numpy as np

# Prepare data for anomaly detection
crime_features = np.array(crime_likelihood).reshape(-1, 1)  # Use crime likelihoods for anomaly detection

# Train Isolation Forest for anomaly detection
isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # Set contamination to 5%
anomaly_labels = isolation_forest.fit_predict(crime_features)  # -1 indicates anomaly, 1 indicates normal

# Categorize points based on likelihood and anomaly detection
crime_colors = []
for i, likelihood in enumerate(crime_likelihood):
    if anomaly_labels[i] == -1:  # Anomalous point
        crime_colors.append('purple')  # Anomaly color
    else:
        crime_colors.append('green' if likelihood < 0.51 else 'red')  # Normal points (low: green, high: red)

# Create a Folium map centered on the dataset's average location
crime_map = folium.Map(location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()], zoom_start=12)

# Add circle markers for each data point based on anomaly detection
for i, (lat, lon) in enumerate(real_data_locations):
    color = crime_colors[i]
    folium.CircleMarker(
        location=(lat, lon),  # Ensure lat and lon are real data points
        radius=1,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"Crime Likelihood: {crime_likelihood[i]:.2f}, Anomalous: {'Yes' if anomaly_labels[i] == -1 else 'No'}"
    ).add_to(crime_map)

# Add a draggable legend
legend_html = """
<div id='legend' style="position: absolute; 
            bottom: 50px; left: 50px; width: 300px; height: auto; 
            background-color: white; z-index: 1000; font-size: 14px; border:2px solid grey; padding: 10px; border-radius: 8px;">
<b>Crime Possibility and Anomaly Legend</b><br>
<span style="color:green;">&#9679;</span> Low Crime Possibility (&lt; 0.5)<br>
<span style="color:red;">&#9679;</span> High Crime Possibility (&ge; 0.5)<br>
<span style="color:purple;">&#9679;</span> Anomalous Data Point
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
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
map_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Anomaly_Detection_Crime_Map_No_Heatmap.html'
crime_map.save(map_path)

print(f"Anomaly detection-based crime map saved to: {map_path}")










 # with highlighted anomaly detection
 
import folium
from sklearn.ensemble import IsolationForest
import numpy as np

# Prepare data for anomaly detection
crime_features = np.array(crime_likelihood).reshape(-1, 1)  # Use crime likelihoods for anomaly detection

# Train Isolation Forest for anomaly detection
isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # Set contamination to 5%
anomaly_labels = isolation_forest.fit_predict(crime_features)  # -1 indicates anomaly, 1 indicates normal

# Categorize points based on likelihood and anomaly detection
crime_colors = []
marker_sizes = []
for i, likelihood in enumerate(crime_likelihood):
    if anomaly_labels[i] == -1:  # Anomalous point
        crime_colors.append('purple')  # Anomaly color
        marker_sizes.append(10)  # Larger size for anomalies
    else:
        if likelihood < 0.51:  # Low crime likelihood
            crime_colors.append('green')
        else:  # High crime likelihood
            crime_colors.append('red')
        marker_sizes.append(6)  # Smaller size for normal points

# Create a Folium map centered on the dataset's average location
crime_map = folium.Map(location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()], zoom_start=12)

# Add circle markers for each data point with emphasis on anomalies
for i, (lat, lon) in enumerate(real_data_locations):
    color = crime_colors[i]
    radius = marker_sizes[i]
    folium.CircleMarker(
        location=(lat, lon),  # Ensure lat and lon are real data points
        radius=2,  # Larger for anomalies, smaller for normal points
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"Crime Likelihood: {crime_likelihood[i]:.2f}, Anomalous: {'Yes' if anomaly_labels[i] == -1 else 'No'}"
    ).add_to(crime_map)

# Add a draggable legend
legend_html = """
<div id='legend' style="position: absolute; 
            bottom: 50px; left: 50px; width: 300px; height: auto; 
            background-color: white; z-index: 1000; font-size: 14px; border:2px solid grey; padding: 10px; border-radius: 8px;">
<b>Crime Possibility and Anomaly Legend</b><br>
<span style="color:green;">&#9679;</span> Low Crime Possibility (&lt; 0.5)<br>
<span style="color:red;">&#9679;</span> High Crime Possibility (&ge; 0.5)<br>
<span style="color:purple; font-size: 18px;">&#9679;</span> Detected Anomaly
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
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
map_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Anomaly_Detection_Highlighted_Map.html'
crime_map.save(map_path)

print(f"Anomaly detection-based crime map with highlighted anomalies saved to: {map_path}")




















