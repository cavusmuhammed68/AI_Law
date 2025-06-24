import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GRU, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Shooting_Incident_Data.csv'  # Adjust the path if necessary
shooting_data = pd.read_csv(file_path)

# Data Preprocessing
# Normalize race columns and replace invalid entries
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].str.upper()
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].str.upper()
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].replace(['(NULL)', 'UNKNOWN'], np.nan)
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].replace(['(NULL)', 'UNKNOWN'], np.nan)

# Drop rows with critical NaN values
shooting_data = shooting_data.dropna(subset=['PERP_RACE', 'VIC_RACE', 'Latitude', 'Longitude', 'OCCUR_DATE', 'OCCUR_TIME'])

# Add time-based features
shooting_data['OCCUR_DATE'] = pd.to_datetime(shooting_data['OCCUR_DATE'])
shooting_data['OCCUR_TIME'] = pd.to_datetime(shooting_data['OCCUR_TIME'], format='%H:%M:%S', errors='coerce')
shooting_data['YEAR'] = shooting_data['OCCUR_DATE'].dt.year
shooting_data['MONTH'] = shooting_data['OCCUR_DATE'].dt.month
shooting_data['DAY'] = shooting_data['OCCUR_DATE'].dt.day
shooting_data['HOUR'] = shooting_data['OCCUR_TIME'].dt.hour

# Scale numeric features
scaler = MinMaxScaler()
numeric_features = ['Latitude', 'Longitude', 'YEAR', 'MONTH', 'DAY', 'HOUR']
scaled_numeric = scaler.fit_transform(shooting_data[numeric_features])

# Encode categorical features
import sklearn
if sklearn.__version__ >= "1.2.0":
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

demographic_features = ['BORO', 'VIC_SEX', 'PERP_SEX']
encoded_demographics = encoder.fit_transform(shooting_data[demographic_features])

# Combine numeric and categorical features
X = np.hstack([scaled_numeric, encoded_demographics])

# Encode target variables
label_encoder_perp = LabelEncoder()
shooting_data['PERP_RACE_ENC'] = label_encoder_perp.fit_transform(shooting_data['PERP_RACE'])

label_encoder_vic = LabelEncoder()
shooting_data['VIC_RACE_ENC'] = label_encoder_vic.fit_transform(shooting_data['VIC_RACE'])

# Define targets
y_perp = shooting_data['PERP_RACE_ENC']
y_vic = shooting_data['VIC_RACE_ENC']
protected_attribute = shooting_data['PERP_RACE_ENC']

# Train-test split
X_train, X_test, y_perp_train, y_perp_test, y_vic_train, y_vic_test, prot_train, prot_test = train_test_split(
    X, y_perp, y_vic, protected_attribute, test_size=0.2, random_state=42
)

# Define the MFR-Net Model
input_layer = Input(shape=(X_train.shape[1],))
reshaped_input = Reshape((X_train.shape[1], 1))(input_layer)
gru_layer = GRU(64, return_sequences=False, dropout=0.3)(reshaped_input)

# Common feature layer
fusion_layer = Dense(128, activation='relu')(gru_layer)
fusion_layer = BatchNormalization()(fusion_layer)
fusion_layer = Dropout(0.3)(fusion_layer)

# Outputs for perpetrator and victim predictions
perp_output = Dense(len(label_encoder_perp.classes_), activation='softmax', name='PERP_OUTPUT')(fusion_layer)
vic_output = Dense(len(label_encoder_vic.classes_), activation='softmax', name='VIC_OUTPUT')(fusion_layer)

# Adversarial output for fairness mitigation
adversary_output = Dense(len(np.unique(protected_attribute)), activation='softmax', name='ADVERSARY_OUTPUT')(fusion_layer)

# Compile the model
model = Model(inputs=input_layer, outputs=[perp_output, vic_output, adversary_output])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'PERP_OUTPUT': 'sparse_categorical_crossentropy',
        'VIC_OUTPUT': 'sparse_categorical_crossentropy',
        'ADVERSARY_OUTPUT': 'sparse_categorical_crossentropy'
    },
    loss_weights={'PERP_OUTPUT': 1.0, 'VIC_OUTPUT': 1.0, 'ADVERSARY_OUTPUT': 0.5},
    metrics={
        'PERP_OUTPUT': 'accuracy',
        'VIC_OUTPUT': 'accuracy',
        'ADVERSARY_OUTPUT': 'accuracy'
    }
)

# Train the model
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
    epochs=50,
    batch_size=64,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]
)

# Evaluate the model
results = model.evaluate(X_test, {
    'PERP_OUTPUT': y_perp_test,
    'VIC_OUTPUT': y_vic_test,
    'ADVERSARY_OUTPUT': prot_test
})
print("Evaluation Results:", results)

# Predictions
y_perp_pred = np.argmax(model.predict(X_test)[0], axis=1)
y_vic_pred = np.argmax(model.predict(X_test)[1], axis=1)

# Fairness Metrics
def calculate_fairness_metrics(y_true, y_pred, prot_attr, privileged_value):
    """
    Calculate fairness metrics: demographic parity, disparate impact, and equal opportunity.
    """
    df = pd.DataFrame({'True': y_true, 'Predicted': y_pred, 'Protected': prot_attr})
    privileged_group = df['Protected'] == privileged_value
    unprivileged_group = ~privileged_group

    privileged_rate = (df[privileged_group]['Predicted'] == df[privileged_group]['True']).mean()
    unprivileged_rate = (df[unprivileged_group]['Predicted'] == df[unprivileged_group]['True']).mean()

    disparate_impact = unprivileged_rate / privileged_rate if privileged_rate > 0 else 0
    equal_opportunity_diff = unprivileged_rate - privileged_rate

    return privileged_rate, unprivileged_rate, disparate_impact, equal_opportunity_diff

# Calculate metrics
privileged_value = label_encoder_perp.transform(['WHITE'])[0]
fairness_metrics = calculate_fairness_metrics(y_perp_test, y_perp_pred, prot_test, privileged_value)

# Visualize Fairness Metrics
metrics = ['Demographic Parity (Privileged)', 'Demographic Parity (Unprivileged)', 
           'Disparate Impact', 'Equal Opportunity Difference']
values = list(fairness_metrics)

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'orange', 'purple', 'green'])
plt.title('Fairness Metrics for Perpetrator Prediction (Post-Mitigation)', fontsize=16)
plt.ylabel('Metric Value')
plt.ylim(0, 1.2)
plt.xticks(rotation=45)
for i, v in enumerate(values):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=12)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

def plot_training_results(history, output_names, save_path=None):
    """
    Plots training and validation accuracy/loss for each output.

    Parameters:
    - history: History object from model.fit()
    - output_names: List of output names (e.g., ['PERP_OUTPUT', 'VIC_OUTPUT', 'ADVERSARY_OUTPUT'])
    - save_path: Optional path to save the plot as an image
    """
    # Extract metrics from the history object
    metrics = history.history

    # Create subplots for each output
    fig, axes = plt.subplots(len(output_names), 2, figsize=(12, 6 * len(output_names)))
    if len(output_names) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for i, output_name in enumerate(output_names):
        # Plot Accuracy
        axes[i][0].plot(metrics[f'{output_name}_accuracy'], label='Training Accuracy')
        axes[i][0].plot(metrics[f'val_{output_name}_accuracy'], label='Validation Accuracy', linestyle='--')
        axes[i][0].set_title(f'{output_name} Accuracy')
        axes[i][0].set_xlabel('Epochs')
        axes[i][0].set_ylabel('Accuracy')
        axes[i][0].legend()
        axes[i][0].grid(True)

        # Plot Loss
        axes[i][1].plot(metrics[f'{output_name}_loss'], label='Training Loss')
        axes[i][1].plot(metrics[f'val_{output_name}_loss'], label='Validation Loss', linestyle='--')
        axes[i][1].set_title(f'{output_name} Loss')
        axes[i][1].set_xlabel('Epochs')
        axes[i][1].set_ylabel('Loss')
        axes[i][1].legend()
        axes[i][1].grid(True)

    # Adjust layout and save if needed
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Plot results
output_names = ['PERP_OUTPUT', 'VIC_OUTPUT', 'ADVERSARY_OUTPUT']
plot_training_results(history, output_names, save_path='training_results.png')
