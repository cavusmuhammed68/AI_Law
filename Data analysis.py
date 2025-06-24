# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:13:39 2024

@author: cavus
"""

# -*- coding: utf-8 -*-
"""
Revised Code with Consistent Coloring, Proper Case, and Font Adjustments
"""

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
file_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\AI and Law Paper 2\Shooting_Incident_Data.csv'
output_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Papers\AI and Law Paper 2'

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

# Race Mapping with Proper Case
race_mapping = {
    "BLACK": "Black",
    "WHITE": "White",
    "ASIAN/PACIFIC ISLANDER": "Asian",
    "AMERICAN INDIAN/ALASKAN NATIVE": "American Indian",
    "WHITE HISPANIC": "White Hispanic",
    "BLACK HISPANIC": "Black Hispanic",
    "UNKNOWN": None
}
shooting_data['PERP_RACE'] = shooting_data['PERP_RACE'].map(race_mapping)
shooting_data['VIC_RACE'] = shooting_data['VIC_RACE'].map(race_mapping)

# Drop rows with unknown or null races
shooting_data = shooting_data.dropna(subset=['PERP_RACE', 'VIC_RACE'])

# Borough Mapping with Proper Case
borough_mapping = {
    "MANHATTAN": "Manhattan",
    "BRONX": "Bronx",
    "BROOKLYN": "Brooklyn",
    "QUEENS": "Queens",
    "STATEN ISLAND": "Staten Island"
}
shooting_data['BORO'] = shooting_data['BORO'].map(borough_mapping)

# Define Consistent Color Palette
consistent_palette = {
    "Black": "#1f77b4",
    "White": "#ff7f0e",
    "Asian": "#2ca02c",
    "American Indian": "#d62728",
    "White Hispanic": "#9467bd",
    "Black Hispanic": "#8c564b"
}

# Bias Analysis: Perpetrator vs. Victim Race
bias_analysis = shooting_data.groupby(['PERP_RACE', 'VIC_RACE']).size().reset_index(name='Incident Count')
bias_pivot = bias_analysis.pivot(index='PERP_RACE', columns='VIC_RACE', values='Incident Count').fillna(0)

# Visualization: Bias Analysis Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    bias_pivot,
    annot=True,
    fmt="g",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Incident Count'}
)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title("Bias Analysis: Perpetrator Race vs Victim Race", fontsize=18)
plt.xlabel("Victim Race", fontsize=16)
plt.ylabel("Perpetrator Race", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_Analysis_Heatmap.png'), dpi=600)
plt.show()

# Normalizing the Bias Analysis Heatmap
bias_pivot_normalized = bias_pivot.div(bias_pivot.sum(axis=1), axis=0) * 100

# Visualization: Normalized Bias Analysis Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    bias_pivot_normalized,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Percentage (%)'}
)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title("Normalized Bias Analysis: Perpetrator Race vs Victim Race (Percentage)", fontsize=18)
plt.xlabel("Victim Race", fontsize=16)
plt.ylabel("Perpetrator Race", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Normalized_Bias_Analysis_Heatmap.png'), dpi=600)
plt.show()

# Visualization 1: Victim Race Distribution by Borough
plt.figure(figsize=(12, 6))
sns.countplot(
    data=shooting_data,
    x='BORO',
    hue='VIC_RACE',
    palette=consistent_palette
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title("Victim Race Distribution by Borough", fontsize=18)
plt.xlabel("Borough", fontsize=16)
plt.ylabel("Incident Count", fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=12)  # Legend title removed
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_By_Borough_Victim_Race.png'), dpi=600)
plt.show()

# Visualization 2: Perpetrator Race Distribution by Borough
plt.figure(figsize=(12, 6))
sns.countplot(
    data=shooting_data,
    x='BORO',
    hue='PERP_RACE',
    palette=consistent_palette
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title("Perpetrator Race Distribution by Borough", fontsize=18)
plt.xlabel("Borough", fontsize=16)
plt.ylabel("Incident Count", fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=12)  # Legend title removed
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_By_Borough_Perpetrator_Race.png'), dpi=600)
plt.show()


# Visualization 3: Geographic Bias (Victim Race)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=shooting_data['Longitude'],
    y=shooting_data['Latitude'],
    hue=shooting_data['VIC_RACE'],
    palette=consistent_palette,
    alpha=0.7
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title("Geographic Distribution of Victim Race", fontsize=18)
plt.xlabel("Longitude", fontsize=16)
plt.ylabel("Latitude", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    loc='upper left',
    bbox_to_anchor=(0, 1),
    fontsize=14
)  # Removed title argument here
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Geographic_Bias_Victim_Race.png'), dpi=600)
plt.show()


# Visualization 4: Geographic Bias (Perpetrator Race)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=shooting_data['Longitude'],
    y=shooting_data['Latitude'],
    hue=shooting_data['PERP_RACE'],
    palette=consistent_palette,
    alpha=0.7
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title("Geographic Distribution of Perpetrator Race", fontsize=18)
plt.xlabel("Longitude", fontsize=16)
plt.ylabel("Latitude", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    loc='upper left',
    bbox_to_anchor=(0, 1),
    fontsize=14
)  # Removed title argument here
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Geographic_Bias_Perpetrator_Race.png'), dpi=600)
plt.show()


print("Updated data import, preprocessing, and visualization completed.")

# To show in the map

import folium
from folium.plugins import HeatMap

# Initialize the map
crime_map = folium.Map(location=[shooting_data['Latitude'].mean(), shooting_data['Longitude'].mean()], zoom_start=11)

# Add Dot Markers for Each Victim Race
for _, row in shooting_data.iterrows():
    color = (
        "#1f77b4" if row['VIC_RACE'] == "Black" else
        "#9467bd" if row['VIC_RACE'] == "White Hispanic" else
        "#8c564b" if row['VIC_RACE'] == "Black Hispanic" else
        "#ff7f0e" if row['VIC_RACE'] == "White" else
        "#d62728" if row['VIC_RACE'] == "American Indian" else
        "gray"
    )
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=0.25,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"Victim Race: {row['VIC_RACE']}, Perpetrator Race: {row['PERP_RACE']}"
    ).add_to(crime_map)

# Add Legend with Draggable Feature
legend_html = """
<div id="legend" style="position: absolute; 
                         top: 50px; right: 50px; 
                         width: 250px; height: auto; 
                         border:2px solid grey; 
                         z-index:9999; 
                         font-size:16px; 
                         background-color:white; 
                         padding: 10px; 
                         border-radius: 5px; 
                         box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
<b>Victim Race Legend</b><br>
&emsp;<span style="color:#1f77b4;">&#9679;</span> Black<br>
&emsp;<span style="color:#9467bd;">&#9679;</span> White Hispanic<br>
&emsp;<span style="color:#8c564b;">&#9679;</span> Black Hispanic<br>
&emsp;<span style="color:#ff7f0e;">&#9679;</span> White<br>
&emsp;<span style="color:#d62728;">&#9679;</span> American Indian<br>
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

# Save Map
output_html_path = os.path.join(output_path, 'Victim_Race_Map.html')
crime_map.save(output_html_path)

print(f"Analysis complete. Map with draggable legend saved to {output_html_path}.")












# Update for Heatmap (Bias Analysis)
plt.figure(figsize=(12, 8))
sns.heatmap(
    bias_pivot.rename(
        columns={"American Indian": "American\nIndian", "Black Hispanic": "Black\nHispanic", "White Hispanic": "White\nHispanic"},
        index={"American Indian": "American\nIndian", "Black Hispanic": "Black\nHispanic", "White Hispanic": "White\nHispanic"}
    ),
    annot=True,
    fmt="g",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Incident Count'},
    annot_kws={"size": 20}  # Set font size for annotations
)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.title("Bias Analysis: Perpetrator Race vs Victim Race", fontsize=18)
plt.xlabel("Victim Race", fontsize=18, labelpad=20)  # Add padding
plt.ylabel("Perpetrator Race", fontsize=18, labelpad=20)  # Add padding
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Bias_Analysis_Heatmap.png'), dpi=600)
plt.show()

# Update for Normalized Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    bias_pivot_normalized.rename(
        columns={"American Indian": "American\nIndian", "Black Hispanic": "Black\nHispanic", "White Hispanic": "White\nHispanic"},
        index={"American Indian": "American\nIndian", "Black Hispanic": "Black\nHispanic", "White Hispanic": "White\nHispanic"}
    ),
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Percentage (%)'},
    annot_kws={"size": 20}  # Set font size for annotations
)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.title("Normalized Bias Analysis: Perpetrator Race vs Victim Race (Percentage)", fontsize=18)
plt.xlabel("Victim Race", fontsize=18, labelpad=20)  # Add padding
plt.ylabel("Perpetrator Race", fontsize=18, labelpad=20)  # Add padding
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Normalized_Bias_Analysis_Heatmap.png'), dpi=600)
plt.show()






















