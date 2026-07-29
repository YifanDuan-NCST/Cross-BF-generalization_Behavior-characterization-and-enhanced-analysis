# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the feature dataset
data = pd.read_csv('BetterFeatureSet.csv', encoding='UTF-8-sig')

# Extract feature data
X = data.values  # Get the numerical part of the feature data
feature_names = data.columns.tolist()  # Get the feature column names

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA algorithm
pca = PCA(n_components=None)  # Initially set to None to compute all principal components
pca.fit(X_scaled)

# Calculate cumulative explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# Determine the minimum number of principal components needed to achieve a cumulative explained variance ratio greater than 85%
n_components = np.argmax(cumulative_explained_variance_ratio > 0.85) + 1

print(f"Number of selected principal components: {n_components}")
print(f"Cumulative explained variance ratio: {cumulative_explained_variance_ratio[n_components - 1]:.2f}")

# Refit PCA with the selected number of principal components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Save the dimensionality-reduced feature data
best_feature_set = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
best_feature_set.to_csv('BestFeatureSet.csv', index=False, encoding='UTF-8-sig')

# Visualize the explained variance ratio of the principal components
import matplotlib.pyplot as plt

# Visualize the explained variance ratio of the principal components
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
plt.axhline(y=0.85, color='r', linestyle='--', label='85% Threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Number of Components')
plt.legend()
plt.grid()
plt.savefig('pca_cumulative_variance.png')  # Save the plot to a file
plt.close()