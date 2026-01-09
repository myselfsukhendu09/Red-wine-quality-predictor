import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# 1. Data Collection
df = pd.read_csv('red-wine-quality.csv', sep=';')

# 2. EDA and Data Preparation
print("Initial Data Info:")
print(df.info())
print("\nStatistical Description:")
print(df.describe())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicates
print(f"\nDuplicate rows found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Data shape after removing duplicates: {df.shape}")

# Outlier Handling (IQR)
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

cols_to_check = df.columns.drop('quality')
df = remove_outliers(df, cols_to_check)
print(f"Data shape after removing outliers: {df.shape}")

# 3. Feature Engineering
# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png')

# Drop low-impact features based on user request
features_to_drop = ['pH', 'fixed acidity', 'citric acid', 'free sulfur dioxide']
df.drop(columns=features_to_drop, inplace=True)
print(f"\nDropped features: {features_to_drop}")

# Target Encoding (Good: >= 7, Bad: < 7)
# Note: In the UCI dataset, quality is usually 3-8. 
# Prompt says Good: >7, but descriptions match >= 7 for this dataset.
print(f"Unique quality values: {df['quality'].unique()}")
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
print(f"Target distribution:\n{df['quality'].value_counts()}")

# Feature Scaling
X = df.drop('quality', axis=1)
y = df['quality']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 4. Data Balancing (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled_df, y)
print(f"Resampled target distribution:\n{pd.Series(y_resampled).value_counts()}")

# 5. Model Development and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SGD': SGDClassifier(random_state=42)
}

print("\nModel Comparison:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc*100:.2f}%")

# Selection and Tuning (XGBoost)
# User request says XGBoost was selected. 
# Tuning parameters mentioned: learning_rate, max_depth, min_child_weight, gamma.
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'n_estimators': [100, 200]
}

print("\nTuning XGBoost...")
grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), 
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Final Evaluation
y_pred_final = best_xgb.predict(X_test)
print("\nFinal Model (XGBoost) Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final)*100:.2f}%")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# 6. Model Pickling
with open('model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and Scaler saved.")
