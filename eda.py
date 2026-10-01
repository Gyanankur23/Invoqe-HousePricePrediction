"""
Exploratory Data Analysis for California Housing Dataset
Generates visualizations and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('data/housing.csv')

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Create output directory
os.makedirs('data', exist_ok=True)

# 1. Correlation Heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('data/correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: data/correlation_heatmap.png")

# 2. Feature Distributions
print("\nGenerating feature distributions...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for i, column in enumerate(df.columns):
    axes[i].hist(df[column], bins=30, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

axes[8].axis('off')
plt.tight_layout()
plt.savefig('data/feature_distributions.png', dpi=150)
plt.close()
print("Saved: data/feature_distributions.png")

# 3. Target Distribution
print("\nGenerating target distribution...")
plt.figure(figsize=(10, 6))
plt.hist(df['MedHouseVal'], bins=30, edgecolor='black')
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('data/target_distribution.png', dpi=150)
plt.close()
print("Saved: data/target_distribution.png")

# 4. Scatter Plots
print("\nGenerating scatter plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(df['MedInc'], df['MedHouseVal'], alpha=0.5)
axes[0, 0].set_xlabel('Median Income')
axes[0, 0].set_ylabel('Median House Value')
axes[0, 0].set_title('Income vs House Value')

axes[0, 1].scatter(df['HouseAge'], df['MedHouseVal'], alpha=0.5)
axes[0, 1].set_xlabel('House Age')
axes[0, 1].set_ylabel('Median House Value')
axes[0, 1].set_title('Age vs House Value')

axes[1, 0].scatter(df['AveRooms'], df['MedHouseVal'], alpha=0.5)
axes[1, 0].set_xlabel('Average Rooms')
axes[1, 0].set_ylabel('Median House Value')
axes[1, 0].set_title('Rooms vs House Value')

axes[1, 1].scatter(df['Population'], df['MedHouseVal'], alpha=0.5)
axes[1, 1].set_xlabel('Population')
axes[1, 1].set_ylabel('Median House Value')
axes[1, 1].set_title('Population vs House Value')

plt.tight_layout()
plt.savefig('data/scatter_plots.png', dpi=150)
plt.close()
print("Saved: data/scatter_plots.png")

# 5. Boxplots
print("\nGenerating boxplots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

df.boxplot(column=['MedInc', 'HouseAge'], ax=axes[0, 0])
axes[0, 0].set_title('Income and Age')

df.boxplot(column=['AveRooms', 'AveBedrms'], ax=axes[0, 1])
axes[0, 1].set_title('Rooms and Bedrooms')

df.boxplot(column=['Population', 'AveOccup'], ax=axes[1, 0])
axes[1, 0].set_title('Population and Occupancy')

df.boxplot(column=['MedHouseVal'], ax=axes[1, 1])
axes[1, 1].set_title('House Value')

plt.tight_layout()
plt.savefig('data/boxplots.png', dpi=150)
plt.close()
print("Saved: data/boxplots.png")

print("\n" + "=" * 50)
print("EDA COMPLETE")
print("=" * 50)
print("Visualizations saved to data/ directory")
