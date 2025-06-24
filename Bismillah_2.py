# Part 1: Data Import, Preprocessing, and Visualization #

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
file_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Shooting_Incident_Data.csv'
output_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2'

# Load dataset
shooting_data = pd.read_csv(file_path)

if shooting_data.empty:
    raise ValueError("Dataset is empty. Please provide a valid dataset.")

# Data Preprocessing
shooting_data['OCCUR_DATE'] = pd.to_datetime(shooting_data['OCCUR_DATE'])
shooting_data['YEAR'] = shooting_data['OCCUR_DATE'].dt.year
shooting_data['MONTH'] = shooting_data['OCCUR_DATE'].dt.month
shooting_data['DAY'] = shooting_data['OCCUR_DATE'].dt.day
shooting_data['HOUR'] = pd.to_datetime(shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour

# Drop rows with missing coordinates
shooting_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Bias Analysis: Perpetrator vs. Victim Race
bias_analysis = shooting_data.groupby(['PERP_RACE', 'VIC_RACE']).size().reset_index(name='Incident Count')
bias_pivot = bias_analysis.pivot(index='PERP_RACE', columns='VIC_RACE', values='Incident Count').fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(
    bias_pivot,
    annot=True,
    fmt="g",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Incident Count'}
)
plt.title("Bias Analysis (Raw Counts): Perpetrator Race vs Victim Race")
plt.xlabel("Victim Race")
plt.ylabel("Perpetrator Race")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_Analysis_Heatmap_Raw.png'), dpi=600)
plt.show()

# Visualization 1: Bias by Borough and Victim Race
plt.figure(figsize=(12, 6))
sns.countplot(data=shooting_data, x='BORO', hue='VIC_RACE', palette='Set2')
plt.title("Victim Race Distribution by Borough")
plt.xlabel("Borough")
plt.ylabel("Incident Count")
plt.xticks(rotation=45)
plt.legend(title="Victim Race", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_By_Borough_Victim_Race.png'), dpi=600)
plt.show()

# Visualization 2: Bias by Borough and Perpetrator Race
plt.figure(figsize=(12, 6))
sns.countplot(data=shooting_data, x='BORO', hue='PERP_RACE', palette='Set1')
plt.title("Perpetrator Race Distribution by Borough")
plt.xlabel("Borough")
plt.ylabel("Incident Count")
plt.xticks(rotation=45)
plt.legend(title="Perpetrator Race", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_By_Borough_Perpetrator_Race.png'), dpi=600)
plt.show()

# Visualization 3: Geographic Bias (Victim Race Heatmap)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=shooting_data['Longitude'],
    y=shooting_data['Latitude'],
    hue=shooting_data['VIC_RACE'],
    palette='tab10',
    alpha=0.7
)
plt.title("Geographic Distribution of Victim Race")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Victim Race", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Geographic_Bias_Victim_Race.png'), dpi=600)
plt.show()

# Visualization 4: Geographic Bias (Perpetrator Race Heatmap)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=shooting_data['Longitude'],
    y=shooting_data['Latitude'],
    hue=shooting_data['PERP_RACE'],
    palette='tab20',
    alpha=0.7
)
plt.title("Geographic Distribution of Perpetrator Race")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Perpetrator Race", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Geographic_Bias_Perpetrator_Race.png'), dpi=600)
plt.show()

# Pivot Data for Heatmap
bias_pivot = bias_analysis.pivot(index='PERP_RACE', columns='VIC_RACE', values='Incident Count').fillna(0)

# Optional Normalization: Convert Raw Counts to Percentages by Row
bias_pivot_normalized = bias_pivot.div(bias_pivot.sum(axis=1), axis=0) * 100

# Plot Heatmap: Raw Counts
plt.figure(figsize=(12, 8))
sns.heatmap(
    bias_pivot,
    annot=True,
    fmt="g",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Incident Count'}
)
plt.title("Bias Analysis (Raw Counts): Perpetrator Race vs Victim Race")
plt.xlabel("Victim Race")
plt.ylabel("Perpetrator Race")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_Analysis_Heatmap_Raw.png'), dpi=600)
plt.show()

# Plot Heatmap: Normalized Values (Percentages)
plt.figure(figsize=(12, 8))
sns.heatmap(
    bias_pivot_normalized,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Percentage'}
)
plt.title("Bias Analysis (Normalized): Perpetrator Race vs Victim Race")
plt.xlabel("Victim Race")
plt.ylabel("Perpetrator Race")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_Analysis_Heatmap_Normalized.png'), dpi=600)
plt.show()


# Feature Engineering: Time of Day
def time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

shooting_data['TIME_OF_DAY'] = shooting_data['HOUR'].apply(time_of_day)
shooting_data.fillna({'PERP_RACE': 'UNKNOWN', 'VIC_RACE': 'UNKNOWN'}, inplace=True)

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
categorical_columns = ['BORO', 'LOC_OF_OCCUR_DESC', 'PERP_RACE', 'VIC_RACE', 'TIME_OF_DAY']
for col in categorical_columns:
    shooting_data[col] = encoder.fit_transform(shooting_data[col])

print("Data import, preprocessing, and enhanced visualization completed.")











# Part 2: Modeling and Classification (Random Forest, Gradient Boosting, and Deep Learning) #

# Part 2a: Advanced Deep Learning Model Building and Training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
import os
import tensorflow as tf

import numpy as np
import pandas as pd

# --- Feature Engineering ---
features = ['BORO', 'LOC_OF_OCCUR_DESC', 'YEAR', 'MONTH', 'VIC_RACE', 'PERP_RACE', 'TIME_OF_DAY']
categorical_features = ['BORO', 'LOC_OF_OCCUR_DESC', 'VIC_RACE', 'PERP_RACE']
numerical_features = ['YEAR', 'MONTH', 'TIME_OF_DAY']

# Preprocessing Features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Apply the preprocessor
X = preprocessor.fit_transform(shooting_data[features])
y = shooting_data['STATISTICAL_MURDER_FLAG'].apply(lambda x: 1 if x else 0)

# Ensure X is dense if not already
if hasattr(X, "todense"):  # Check if X is a sparse matrix
    X = X.todense()

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# --- Residual Model ---
def build_residual_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    residual = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    residual = BatchNormalization()(residual)
    residual = Dropout(0.4)(residual)
    x = Add()([x, residual])
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Residual_Model")

# --- Attention Model ---
def build_attention_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    query = Dense(128)(x)
    key = Dense(128)(x)
    value = Dense(128)(x)
    attention_output = Attention()([query, key, value])
    attention_output = Dense(256)(attention_output)
    x = Add()([x, attention_output])
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Attention_Model")

# --- Training and Evaluation ---
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, output_path):
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    model.save(os.path.join(output_path, f'{model_name}.h5'))
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(
        os.path.join(output_path, f"{model_name}_Classification_Report.csv")
    )
    return accuracy, history

# Output Path
output_path = "./model_output"
os.makedirs(output_path, exist_ok=True)

# Train Models
residual_model = build_residual_model(input_shape=X_train.shape[1])
residual_accuracy, residual_history = train_and_evaluate_model(
    residual_model, X_train, X_test, y_train, y_test, "Residual_Model", output_path
)

attention_model = build_attention_model(input_shape=X_train.shape[1])
attention_accuracy, attention_history = train_and_evaluate_model(
    attention_model, X_train, X_test, y_train, y_test, "Attention_Model", output_path
)


# Part 2b: Ensemble Predictions, Visualizations, and Evaluations

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ensemble Predictions ---
y_pred_residual = residual_model.predict(X_test).flatten()
y_pred_attention = attention_model.predict(X_test).flatten()

# Weighted Ensemble
ensemble_predictions = (0.6 * y_pred_residual + 0.4 * y_pred_attention)
ensemble_predictions = (ensemble_predictions > 0.5).astype(int)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")

pd.DataFrame(classification_report(y_test, ensemble_predictions, output_dict=True)).transpose().to_csv(
    os.path.join(output_path, "Ensemble_Classification_Report.csv")
)

# --- Visualizations ---
model_comparison = pd.DataFrame({
    'Model': ['Residual Model', 'Attention Model', 'Ensemble Model'],
    'Accuracy': [residual_accuracy, attention_accuracy, ensemble_accuracy]
})

sns.barplot(data=model_comparison, x='Accuracy', y='Model', palette='Blues_d')
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Model_Accuracy_Comparison.png'))
plt.show()

conf_matrix = confusion_matrix(y_test, ensemble_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Non-Murder', 'Murder'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: Ensemble Model")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Confusion_Matrix.png'))
plt.show()




# Part 2c: Reinforcement Learning and Hybrid Models

from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Parameters ---
state_space = X_resampled  # Features as states
action_space = [0, 1]  # Predict non-murder (0) or murder (1)

# Hyperparameters
learning_rate = 0.0005
discount_factor = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
batch_size = 128
memory_size = 2000
num_episodes = 500

# --- Reward Function ---
def get_reward(true_label, predicted_label):
    """Assign a reward based on prediction correctness."""
    return 1 if true_label == predicted_label else -1

# --- Build Hybrid Model ---
def build_hybrid_model(input_shape, action_size):
    """Builds a hybrid model for SL and RL."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(action_size, activation='linear')  # Output Q-values for actions
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])
    return model

# Initialize Hybrid Model
input_shape = X_train.shape[1]
action_size = len(action_space)
hybrid_model = build_hybrid_model(input_shape, action_size)

# Replay Memory
memory = deque(maxlen=memory_size)

# --- Supervised Learning (SL) Pretraining ---
print("\nPretraining Hybrid Model with Supervised Learning...")
hybrid_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
hybrid_model.fit(
    X_train, tf.keras.utils.to_categorical(y_train, num_classes=2),
    validation_data=(X_test, tf.keras.utils.to_categorical(y_test, num_classes=2)),
    epochs=50,
    batch_size=64,
    verbose=1
)

# Evaluate SL Model
sl_predictions = np.argmax(hybrid_model.predict(X_test), axis=1)
sl_accuracy = accuracy_score(y_test, sl_predictions)
print(f"Supervised Learning Accuracy: {sl_accuracy:.4f}")

# --- Reinforcement Learning (RL) Fine-Tuning ---
print("\nFine-tuning Hybrid Model with Reinforcement Learning...")
rewards_all_episodes = []

for episode in range(num_episodes):
    state_idx = random.randint(0, len(state_space) - 1)  # Start at a random state
    state = state_space[state_idx]
    total_reward = 0
    done = False

    while not done:
        # Epsilon-Greedy Action Selection
        if np.random.rand() < epsilon:
            action = random.choice(action_space)  # Explore
        else:
            q_values = hybrid_model.predict(np.expand_dims(state, axis=0), verbose=0)
            action = np.argmax(q_values)  # Exploit

        # Get reward and next state
        true_label = y_resampled[state_idx]
        reward = get_reward(true_label, action)
        total_reward += reward

        next_state_idx = (state_idx + 1) % len(state_space)
        next_state = state_space[next_state_idx]

        # Store transition in memory
        memory.append((state, action, reward, next_state))

        # Move to the next state
        state_idx = next_state_idx
        state = next_state

        # End episode if last state is reached
        if state_idx == len(state_space) - 1:
            done = True

        # Experience Replay
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)

            states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])

            q_values_next = hybrid_model.predict(next_states, verbose=0)
            q_values_target = hybrid_model.predict(states, verbose=0)

            for i in range(batch_size):
                target = rewards[i] + discount_factor * np.max(q_values_next[i])
                q_values_target[i][actions[i]] = target

            hybrid_model.train_on_batch(states, q_values_target)

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_all_episodes.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

# --- Evaluate Hybrid Model ---
print("\nEvaluating Hybrid Model...")
hybrid_predictions = np.argmax(hybrid_model.predict(X_test), axis=1)
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
print(f"Hybrid Model Accuracy: {hybrid_accuracy:.4f}")

# Save Hybrid Model
hybrid_model.save(os.path.join(output_path, 'Optimized_Hybrid_Model.h5'))

# --- Visualization: RL Training Rewards ---
plt.figure(figsize=(10, 6))
plt.plot(range(num_episodes), rewards_all_episodes, color='blue')
plt.title("RL-SL Hybrid Training Rewards per Episode")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Hybrid_Training_Rewards.png'), dpi=600)
plt.show()

print(f"Hybrid Model training and evaluation completed. Accuracy: {hybrid_accuracy:.4f}")


















# Part 3: Anomaly Detection and Unsupervised Learning #

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

# === Anomaly Detection ===
print("\n=== Anomaly Detection ===")

# Prepare features for anomaly detection
anomaly_features = ['Latitude', 'Longitude', 'YEAR', 'MONTH', 'VIC_RACE', 'PERP_RACE']
anomaly_data = shooting_data[anomaly_features]

# Normalize features
scaler = MinMaxScaler()
normalized_anomaly_data = scaler.fit_transform(anomaly_data)

# 1. Isolation Forest
print("Running Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(normalized_anomaly_data)
shooting_data['Isolation_Anomaly'] = (iso_labels == -1).astype(int)

# 2. One-Class SVM
print("Running One-Class SVM...")
one_class_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm_labels = one_class_svm.fit_predict(normalized_anomaly_data)
shooting_data['SVM_Anomaly'] = (svm_labels == -1).astype(int)

# 3. Local Outlier Factor
print("Running Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(normalized_anomaly_data)
shooting_data['LOF_Anomaly'] = (lof_labels == -1).astype(int)

# Combine Results for Consensus
shooting_data['Consensus_Anomaly'] = (
    shooting_data['Isolation_Anomaly'] +
    shooting_data['SVM_Anomaly'] +
    shooting_data['LOF_Anomaly']
)

# Majority voting (consensus anomaly detection)
shooting_data['Final_Anomaly'] = (shooting_data['Consensus_Anomaly'] >= 2).astype(int)

# Analyze and Visualize Anomalies
total_anomalies = shooting_data['Final_Anomaly'].sum()
print(f"Total anomalies detected: {total_anomalies}")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=shooting_data['Latitude'],
    y=shooting_data['Longitude'],
    hue=shooting_data['Final_Anomaly'],
    palette={0: 'blue', 1: 'red'}
)
plt.title("Anomaly Detection: Normal vs. Anomalous Incidents")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend(title="Anomaly", loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Anomaly_Detection_Map.png'), dpi=600)
plt.show()

# Save Anomaly Results
anomaly_results_path = os.path.join(output_path, 'Anomaly_Detection_Results.csv')
shooting_data.to_csv(anomaly_results_path, index=False)
print(f"Anomaly detection results saved to: {anomaly_results_path}")

# === Unsupervised Learning ===
print("\n=== Unsupervised Learning ===")

# Prepare data for clustering
unsupervised_features = ['Latitude', 'Longitude', 'YEAR', 'MONTH', 'VIC_RACE', 'PERP_RACE', 'TIME_OF_DAY']
unsupervised_data = shooting_data[unsupervised_features]

# Normalize features
normalized_unsupervised_data = scaler.fit_transform(unsupervised_data)

# 1. Clustering with KMeans
print("Running KMeans Clustering...")
range_n_clusters = range(2, 11)
silhouette_scores = []

# Find optimal clusters using Silhouette Score
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_unsupervised_data)
    silhouette_avg = silhouette_score(normalized_unsupervised_data, cluster_labels)
    silhouette_scores.append((n_clusters, silhouette_avg))
    print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")

# Select optimal number of clusters
optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"Optimal number of clusters: {optimal_clusters}")

# Apply KMeans with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
shooting_data['KMeans_Cluster'] = kmeans.fit_predict(normalized_unsupervised_data)

# 2. Density-Based Clustering (DBSCAN)
print("Running DBSCAN...")
dbscan = DBSCAN(eps=0.2, min_samples=10)
shooting_data['DBSCAN_Cluster'] = dbscan.fit_predict(normalized_unsupervised_data)

# 3. Dimensionality Reduction with PCA
print("Running PCA for Dimensionality Reduction...")
pca = PCA(n_components=2)
pca_components = pca.fit_transform(normalized_unsupervised_data)
shooting_data['PCA_1'] = pca_components[:, 0]
shooting_data['PCA_2'] = pca_components[:, 1]

# Visualize Clusters from PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=shooting_data['PCA_1'],
    y=shooting_data['PCA_2'],
    hue=shooting_data['KMeans_Cluster'],
    palette='viridis',
    style=shooting_data['DBSCAN_Cluster']
)
plt.title("Clusters Visualized in PCA Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="KMeans Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'PCA_Cluster_Visualization.png'), dpi=600)
plt.show()

# 4. Dimensionality Reduction with t-SNE
print("Running t-SNE for Visualization...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_components = tsne.fit_transform(normalized_unsupervised_data)
shooting_data['tSNE_1'] = tsne_components[:, 0]
shooting_data['tSNE_2'] = tsne_components[:, 1]

# Visualize Clusters from t-SNE
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=shooting_data['tSNE_1'],
    y=shooting_data['tSNE_2'],
    hue=shooting_data['KMeans_Cluster'],
    palette='viridis',
    style=shooting_data['DBSCAN_Cluster']
)
plt.title("Clusters Visualized in t-SNE Space")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="KMeans Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'tSNE_Cluster_Visualization.png'), dpi=600)
plt.show()

# Use Clusters as Features
shooting_data['Cluster_Label'] = shooting_data['KMeans_Cluster']
features = ['BORO', 'LOC_OF_OCCUR_DESC', 'YEAR', 'MONTH', 'VIC_RACE', 'PERP_RACE', 'TIME_OF_DAY', 'Cluster_Label']
X = shooting_data[features]
y = shooting_data['STATISTICAL_MURDER_FLAG'].apply(lambda x: 1 if x else 0)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train Random Forest with Cluster Feature
print("\nTraining Random Forest with Clusters as Features...")
rf_model_with_clusters = RandomForestClassifier(random_state=42)
rf_model_with_clusters.fit(X_train, y_train)

# Evaluate Random Forest with Cluster Feature
y_pred_rf_clusters = rf_model_with_clusters.predict(X_test)
rf_clusters_accuracy = accuracy_score(y_test, y_pred_rf_clusters)
print(f"Random Forest with Clusters Accuracy: {rf_clusters_accuracy:.4f}")

# Save Results
report_rf_clusters = classification_report(y_test, y_pred_rf_clusters, output_dict=True)
report_rf_clusters_df = pd.DataFrame(report_rf_clusters).transpose()
report_rf_clusters_path = os.path.join(output_path, 'Random_Forest_Cluster_Classification_Report.csv')
report_rf_clusters_df.to_csv(report_rf_clusters_path)
print(f"Random Forest with Clusters classification report saved to {report_rf_clusters_path}")
