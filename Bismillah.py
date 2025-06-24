import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import os

# Set file paths
file_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Shooting_Incident_Data.csv'
output_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2'

# Load the dataset
shooting_data = pd.read_csv(file_path)

# Data Preprocessing
shooting_data['OCCUR_DATE'] = pd.to_datetime(shooting_data['OCCUR_DATE'])
shooting_data['YEAR'] = shooting_data['OCCUR_DATE'].dt.year
shooting_data['MONTH'] = shooting_data['OCCUR_DATE'].dt.month
shooting_data['DAY'] = shooting_data['OCCUR_DATE'].dt.day
shooting_data['HOUR'] = pd.to_datetime(shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour

# Remove rows with missing coordinates
shooting_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Feature Engineering: Add Time of Day
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
shooting_data['TIME_OF_DAY'] = LabelEncoder().fit_transform(shooting_data['TIME_OF_DAY'])

# Fill missing values
shooting_data.fillna({'LOC_OF_OCCUR_DESC': 'UNKNOWN', 'PERP_RACE': 'UNKNOWN', 'VIC_RACE': 'UNKNOWN'}, inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
shooting_data['BORO'] = encoder.fit_transform(shooting_data['BORO'])
shooting_data['LOC_OF_OCCUR_DESC'] = encoder.fit_transform(shooting_data['LOC_OF_OCCUR_DESC'])
shooting_data['PERP_RACE'] = encoder.fit_transform(shooting_data['PERP_RACE'])
shooting_data['VIC_RACE'] = encoder.fit_transform(shooting_data['VIC_RACE'])

# Create target variable
shooting_data['MURDER_FLAG'] = shooting_data['STATISTICAL_MURDER_FLAG'].apply(lambda x: 1 if x else 0)

# Features and Target
features = ['BORO', 'LOC_OF_OCCUR_DESC', 'PRECINCT', 'YEAR', 'MONTH', 'DAY', 'VIC_RACE', 'PERP_RACE', 'TIME_OF_DAY']
X = shooting_data[features]
y = shooting_data['MURDER_FLAG']

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Normalize Features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_resampled)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_resampled, test_size=0.3, random_state=42)

# Random Forest Model with Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Save Classification Report
classification_report_result = classification_report(y_test, y_pred, output_dict=True)
classification_df = pd.DataFrame(classification_report_result).transpose()
classification_df.to_csv(os.path.join(output_path, 'Classification_Report.csv'))

# Create Prediction Categories
categories = pd.cut(y_proba, bins=[0, 0.33, 0.66, 1], labels=["Low Crime Possible", "Middle Crime Possible", "High Crime Possible"])
category_counts = categories.value_counts()

# Plot Prediction Categories
plt.figure(figsize=(8, 6))
category_counts.plot(kind='bar', color=['green', 'blue', 'red'])
plt.title("Prediction Categories")
plt.xlabel("Crime Probability Categories")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Prediction_Categories.png'), dpi=600)
plt.show()

# Add Predictions to DataFrame
original_indices = X.iloc[X_test.shape[0] * -1:].index  # Recover indices of test data
prediction_data = pd.DataFrame({
    'Latitude': shooting_data.loc[original_indices, 'Latitude'].values,
    'Longitude': shooting_data.loc[original_indices, 'Longitude'].values,
    'Probability': y_proba,
    'Category': categories
}).dropna()

# Create HeatMap with Legend
crime_map = folium.Map(location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()], zoom_start=11)

# Add Dot Markers
for _, row in prediction_data.iterrows():
    color = 'green' if row['Category'] == "Low Crime Possible" else 'blue' if row['Category'] == "Middle Crime Possible" else 'red'
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=1,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"Category: {row['Category']}, Probability: {row['Probability']:.2f}"
    ).add_to(crime_map)

# Add Legend with Draggable Feature
legend_html = """
<div id="legend" style="position: absolute; 
                         bottom: 50px; left: 50px; 
                         width: 250px; height: 150px; 
                         border:2px solid grey; 
                         z-index:9999; 
                         font-size:16px; 
                         background-color:white; 
                         padding: 10px; 
                         border-radius: 5px; 
                         cursor: move;">
<b>Crime Probability Legend</b><br>
&emsp;<span style="color:green;">&#9679;</span> Low Crime Possible<br>
&emsp;<span style="color:blue;">&#9679;</span> Middle Crime Possible<br>
&emsp;<span style="color:red;">&#9679;</span> High Crime Possible<br>
</div>

<script>
    const legend = document.getElementById('legend');
    let isMouseDown = false, offsetX, offsetY;

    legend.addEventListener('mousedown', function(e) {
        isMouseDown = true;
        offsetX = e.clientX - legend.offsetLeft;
        offsetY = e.clientY - legend.offsetTop;
    });

    document.addEventListener('mouseup', function() {
        isMouseDown = false;
    });

    document.addEventListener('mousemove', function(e) {
        if (isMouseDown) {
            legend.style.left = (e.clientX - offsetX) + 'px';
            legend.style.top = (e.clientY - offsetY) + 'px';
        }
    });
</script>
"""
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save Map
crime_map.save(os.path.join(output_path, 'Crime_Prediction_Categories_Map.html'))

print(f"Analysis complete. Results and map saved in {output_path}")




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# Set file paths
file_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Shooting_Incident_Data.csv'
output_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2'

# Load the dataset
shooting_data = pd.read_csv(file_path)

# Data Preprocessing
shooting_data['OCCUR_DATE'] = pd.to_datetime(shooting_data['OCCUR_DATE'])
shooting_data['YEAR'] = shooting_data['OCCUR_DATE'].dt.year
shooting_data['MONTH'] = shooting_data['OCCUR_DATE'].dt.month
shooting_data['DAY'] = shooting_data['OCCUR_DATE'].dt.day
shooting_data['HOUR'] = pd.to_datetime(shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour

# Remove rows with missing coordinates
shooting_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Feature Engineering: Add Time of Day
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
shooting_data['TIME_OF_DAY'] = LabelEncoder().fit_transform(shooting_data['TIME_OF_DAY'])

# Fill missing values
shooting_data.fillna({'LOC_OF_OCCUR_DESC': 'UNKNOWN', 'PERP_RACE': 'UNKNOWN', 'VIC_RACE': 'UNKNOWN'}, inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
shooting_data['BORO'] = encoder.fit_transform(shooting_data['BORO'])
shooting_data['LOC_OF_OCCUR_DESC'] = encoder.fit_transform(shooting_data['LOC_OF_OCCUR_DESC'])
shooting_data['PERP_RACE'] = encoder.fit_transform(shooting_data['PERP_RACE'])
shooting_data['VIC_RACE'] = encoder.fit_transform(shooting_data['VIC_RACE'])

# Create target variable
shooting_data['MURDER_FLAG'] = shooting_data['STATISTICAL_MURDER_FLAG'].apply(lambda x: 1 if x else 0)

# Features and Target
features = ['BORO', 'LOC_OF_OCCUR_DESC', 'PRECINCT', 'YEAR', 'MONTH', 'DAY', 'VIC_RACE', 'PERP_RACE', 'TIME_OF_DAY']
X = shooting_data[features]
y = shooting_data['MURDER_FLAG']

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Normalize Features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_resampled)

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.3, random_state=42)

# Random Forest Model
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Deep Learning (CNN-RNN) Model for Sequential Patterns
X_cnn = X_resampled.values.reshape(X_resampled.shape[0], X_resampled.shape[1], 1)
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_cnn, y_resampled, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate Models
y_pred = random_forest.predict(X_test)
y_proba_rf = random_forest.predict_proba(X_test)[:, 1]
y_pred_dl = (model.predict(X_cnn) > 0.5).astype(int)

# K-Means Clustering for Geospatial Analysis
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Visualization of Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Geospatial Clusters of Shooting Incidents")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.savefig(os.path.join(output_path, 'Geospatial_Clusters.png'), dpi=600)
plt.show()

# Bias Analysis: Check Distribution of Perpetrator and Victim Race
print("Distribution of Perpetrator Races:\n", shooting_data['PERP_RACE'].value_counts())
print("Distribution of Victim Races:\n", shooting_data['VIC_RACE'].value_counts())

# Grouping Data for Bias Analysis
bias_analysis = shooting_data.groupby(['PERP_RACE', 'VIC_RACE']).size().reset_index(name='Incident Count')

# Handle Missing Values
bias_analysis['PERP_RACE'] = bias_analysis['PERP_RACE'].fillna('UNKNOWN')
bias_analysis['VIC_RACE'] = bias_analysis['VIC_RACE'].fillna('UNKNOWN')

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



# Real-Time Crime Mapping
crime_map = folium.Map(location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()], zoom_start=11)
for lat, lon, cluster in zip(shooting_data['Latitude'], shooting_data['Longitude'], clusters):
    folium.CircleMarker(location=[lat, lon], radius=5, color='blue', fill=True, fill_opacity=0.5,
                        popup=f"Cluster: {cluster}").add_to(crime_map)
crime_map.save(os.path.join(output_path, 'Crime_Clusters_Map.html'))

print("Results saved successfully.")













import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2\Shooting_Incident_Data.csv'
output_path = r'C:\Users\cavus\Desktop\AI and Law Paper 2'

data = pd.read_csv(file_path)

# Convert dates to datetime for easier manipulation
data['OCCUR_DATE'] = pd.to_datetime(data['OCCUR_DATE'])

# 1. Trend of incidents over time
incident_trend = data.groupby(data['OCCUR_DATE'].dt.to_period('M')).size()
incident_trend.index = incident_trend.index.to_timestamp()

plt.figure(figsize=(12, 6))
plt.plot(incident_trend, marker='o')
plt.title('Monthly Trend of Shooting Incidents')
plt.xlabel('Month-Year')
plt.ylabel('Number of Incidents')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_path}/Monthly_Trend_Shooting_Incidents.png', dpi=600)
plt.show()

# 2. Distribution of incidents by borough
borough_counts = data['BORO'].value_counts()

plt.figure(figsize=(8, 6))
borough_counts.plot(kind='bar')
plt.title('Shooting Incidents by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Shooting_Incidents_By_Borough.png', dpi=600)
plt.show()

# 3. Distribution of incidents by location type
location_counts = data['LOC_OF_OCCUR_DESC'].value_counts()

plt.figure(figsize=(8, 6))
location_counts.plot(kind='bar')
plt.title('Incidents by Location Type')
plt.xlabel('Location Type')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Incidents_By_Location_Type.png', dpi=600)
plt.show()

# 4. Victim demographics (age groups)
age_group_counts = data['VIC_AGE_GROUP'].value_counts()

plt.figure(figsize=(8, 6))
age_group_counts.plot(kind='bar')
plt.title('Victim Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Number of Victims')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Age_Group_Distribution.png', dpi=600)
plt.show()

# 5. Correlation heatmap for numeric variables
plt.figure(figsize=(10, 8))
sns.heatmap(data[['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Coordinates')
plt.tight_layout()
plt.savefig(f'{output_path}/Correlation_Heatmap_Coordinates.png', dpi=600)
plt.show()










import matplotlib.pyplot as plt
import seaborn as sns

# 1. Gender distribution
gender_counts = data['VIC_SEX'].value_counts()

plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar')
plt.title('Victim Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Victims')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Gender_Distribution.png', dpi=600)
plt.show()

# 2. Race distribution
race_counts = data['VIC_RACE'].value_counts()

plt.figure(figsize=(10, 6))
race_counts.plot(kind='bar')
plt.title('Victim Race Distribution')
plt.xlabel('Race')
plt.ylabel('Number of Victims')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Race_Distribution.png', dpi=600)
plt.show()

# 3. Age Group vs. Gender
age_gender_counts = data.groupby(['VIC_AGE_GROUP', 'VIC_SEX']).size().unstack()

plt.figure(figsize=(12, 6))
age_gender_counts.plot(kind='bar', stacked=True)
plt.title('Victim Age Group by Gender')
plt.xlabel('Age Group')
plt.ylabel('Number of Victims')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Age_Group_By_Gender.png', dpi=600)
plt.show()

# 4. Age Group vs. Race
age_race_counts = data.groupby(['VIC_AGE_GROUP', 'VIC_RACE']).size().unstack()

plt.figure(figsize=(12, 6))
age_race_counts.plot(kind='bar', stacked=True)
plt.title('Victim Age Group by Race')
plt.xlabel('Age Group')
plt.ylabel('Number of Victims')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Age_Group_By_Race.png', dpi=600)
plt.show()

# 5. Gender and Race Combined
gender_race_counts = data.groupby(['VIC_SEX', 'VIC_RACE']).size().unstack()

plt.figure(figsize=(12, 6))
gender_race_counts.plot(kind='bar', stacked=True)
plt.title('Victim Gender by Race')
plt.xlabel('Gender')
plt.ylabel('Number of Victims')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Gender_By_Race.png', dpi=600)
plt.show()








import matplotlib.pyplot as plt
import seaborn as sns

# 1. Overall Distribution of Victim Races
race_counts = data['VIC_RACE'].value_counts()

plt.figure(figsize=(8, 6))
race_counts.plot(kind='bar')
plt.title('Overall Victim Race Distribution')
plt.xlabel('Race')
plt.ylabel('Number of Victims')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Overall_Victim_Race_Distribution.png', dpi=600)
plt.show()

# 2. Race Trends Over Time
data['OCCUR_YEAR'] = data['OCCUR_DATE'].dt.year
race_trends = data.groupby(['OCCUR_YEAR', 'VIC_RACE']).size().unstack()

plt.figure(figsize=(12, 6))
race_trends.plot(marker='o')
plt.title('Victim Race Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Victims')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Race_Trends_Over_Time.png', dpi=600)
plt.show()

# 3. Race Distribution by Borough
race_borough = data.groupby(['BORO', 'VIC_RACE']).size().unstack()

plt.figure(figsize=(12, 6))
race_borough.plot(kind='bar', stacked=True)
plt.title('Victim Race Distribution by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Victims')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Race_Distribution_By_Borough.png', dpi=600)
plt.show()

# 4. Race Distribution by Gender
race_gender = data.groupby(['VIC_SEX', 'VIC_RACE']).size().unstack()

plt.figure(figsize=(10, 6))
race_gender.plot(kind='bar', stacked=True)
plt.title('Victim Race Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Victims')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Race_Distribution_By_Gender.png', dpi=600)
plt.show()

# 5. Race Distribution by Age Group
race_age_group = data.groupby(['VIC_AGE_GROUP', 'VIC_RACE']).size().unstack()

plt.figure(figsize=(12, 6))
race_age_group.plot(kind='bar', stacked=True)
plt.title('Victim Race Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Victims')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Race_Distribution_By_Age_Group.png', dpi=600)
plt.show()






import matplotlib.pyplot as plt
import seaborn as sns

# 1. Incident Counts by Borough
borough_counts = data['BORO'].value_counts()

plt.figure(figsize=(8, 6))
borough_counts.plot(kind='bar', color='skyblue')
plt.title('Incident Counts by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Incident_Counts_By_Borough.png', dpi=600)
plt.show()

# 2. Incident Trends Over Time by Borough
borough_trends = data.groupby([data['OCCUR_DATE'].dt.to_period('M'), 'BORO']).size().unstack()

plt.figure(figsize=(12, 6))
borough_trends.plot(marker='o')
plt.title('Incident Trends Over Time by Borough')
plt.xlabel('Month-Year')
plt.ylabel('Number of Incidents')
plt.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_path}/Incident_Trends_By_Borough.png', dpi=600)
plt.show()

# 3. Incident Location Types by Borough
location_borough = data.groupby(['BORO', 'LOC_OF_OCCUR_DESC']).size().unstack()

plt.figure(figsize=(12, 6))
location_borough.plot(kind='bar', stacked=True)
plt.title('Incident Location Types by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Incidents')
plt.legend(title='Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Incident_Location_Types_By_Borough.png', dpi=600)
plt.show()

# 4. Victim Demographics by Borough
# Victim Gender by Borough
gender_borough = data.groupby(['BORO', 'VIC_SEX']).size().unstack()

plt.figure(figsize=(10, 6))
gender_borough.plot(kind='bar', stacked=True, color=['lightblue', 'salmon'])
plt.title('Victim Gender Distribution by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Victims')
plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Gender_By_Borough.png', dpi=600)
plt.show()

# Victim Race by Borough
race_borough = data.groupby(['BORO', 'VIC_RACE']).size().unstack()

plt.figure(figsize=(12, 6))
race_borough.plot(kind='bar', stacked=True)
plt.title('Victim Race Distribution by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Victims')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_path}/Victim_Race_By_Borough.png', dpi=600)
plt.show()











