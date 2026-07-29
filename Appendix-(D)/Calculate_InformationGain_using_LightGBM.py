# -*- coding: utf-8 -*-
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data
data = pd.read_csv('Data.csv', encoding='UTF-8-sig')

# Assume the last column is the classification column, and the rest are parameter columns
X = data.iloc[:, :-1]  # Parameter columns
y = data.iloc[:, -1]   # Classification column (labeled as "1", "2", etc.)

# Convert the classification column to integer type starting from 0
y = y.astype(int)
y = y - min(y)  # Convert labels to start from 0 if they originally start from 1
print(X)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM training and validation datasets
train_dataset = lgb.Dataset(X_train, label=y_train)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

# Set LightGBM parameters
num_class = len(set(y))  # Number of classes
params = {
    'objective': 'multiclass',  # Multiclass classification task
    'num_class': num_class,     # Number of classes
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 20,
    'num_leaves': 10,
    'min_data_in_leaf': 5,
    'feature_fraction': 0.8,
    'min_gain_to_split': 0.0  # Allow splitting even if the information gain is <= 0
}

# Train the model
model = lgb.train(params, train_dataset, num_boost_round=100, valid_sets=[val_dataset])

# Get feature names and information gain values
feature_names = X.columns.tolist()
feature_importance = model.feature_importance(importance_type='gain')

# Normalize the information gain to percentage
total_gain = sum(feature_importance)
if total_gain != 0:
    feature_importance_percentage = (feature_importance / total_gain) * 100
else:
    feature_importance_percentage = np.zeros_like(feature_importance)

# Calculate information gain percentage for each class
class_importance = {cls: [] for cls in sorted(set(y))}
for idx, feature in enumerate(feature_names):
    for cls in class_importance:
        # Create a training set containing only the current class data
        class_data = X[y == cls]
        class_dataset = lgb.Dataset(class_data, label=[cls]*len(class_data))
        # Train a model and get feature importance
        class_model = lgb.train(params, class_dataset, num_boost_round=100)
        # Get the information gain for the current feature and convert to percentage
        class_gain = class_model.feature_importance(importance_type='gain')[idx]
        total_class_gain = sum(class_model.feature_importance(importance_type='gain'))
        if total_class_gain != 0:
            class_importance[cls].append((class_gain / total_class_gain) * 100)
        else:
            class_importance[cls].append(0)  # Avoid division by zero

# Plot the information gain percentage for each class
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = range(len(feature_names))
colors = ['b', 'g', 'r', 'c', 'm']  # Adjust colors based on the number of classes

for i, cls in enumerate(sorted(class_importance.keys())):
    plt.bar([idx + i * bar_width for idx in index], class_importance[cls], bar_width, label=f'Class {cls}', color=colors[i % len(colors)])

plt.xlabel('Features')
plt.ylabel('Information Gain (%)')  # Add percentage unit to the y-axis label
plt.title('Information Gain by Feature and Class')
plt.xticks([idx + bar_width * (len(class_importance) - 1) / 2 for idx in index], feature_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('information_gain.png')  # 保存图表到文件
plt.close()