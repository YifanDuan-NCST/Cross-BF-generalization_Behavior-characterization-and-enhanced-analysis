# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Read the feature dataset
data = pd.read_csv('FeatureData.csv', encoding='UTF-8-sig')

# Extract feature data
X = data.values  # Get the numerical part of the feature data
feature_names = data.columns.tolist()  # Get the feature column names

# Data preprocessing: Check and handle NaN or infinite values
if np.isnan(X).any() or np.isinf(X).any():
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Data has been standardized.")
else:
    print("No data preprocessing required.")

# Define the objective function for optimization
def objective(weights, features):
    weighted_features = np.dot(features, weights)
    return -np.var(weighted_features)

# Define the optimization constraints
constraints = (
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
)

# Define the bounds for the weights
bounds = tuple((0, 1) for _ in range(X.shape[1]))

# Try different initial weights
np.random.seed(42)
initial_weights = np.random.dirichlet(np.ones(X.shape[1]), size=1)[0]

# Perform the optimization
result = minimize(
    objective,
    initial_weights,
    args=(X,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
)

# Output the optimization result
if result.success:
    print("Optimization succeeded!")
else:
    print("Optimization failed! Try different initial weights or optimization method.")

optimized_weights = result.x

# Display the optimized weights
weight_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': optimized_weights
}).sort_values(by='Weight', ascending=False)

print("\nOptimized weights:")
print(weight_df)

# Apply the optimized weights to fuse the features
fused_feature = np.dot(X, optimized_weights)

fused_feature_df = pd.DataFrame({
    'Fused_Feature': fused_feature
})

print("\nFused feature:")
print(fused_feature_df.head())

# Save the results to CSV files
weight_df.to_csv('Optimized_Weights.csv', index=False)
fused_feature_df.to_csv('Fused_Features.csv', index=False)