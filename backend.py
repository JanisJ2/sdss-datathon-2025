import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import xgboost as xgb
import shap
from sklearn.inspection import PartialDependenceDisplay

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

from xgboost import XGBRegressor

# Data Loading & Cleaning
df = pd.read_csv("real-estate-data-with-neighbourhood.csv")

# drop the unnecessary columns
df = df.drop(columns=['id_'])
df = df.drop(columns=['exposure'])
df = df.drop(columns=['parking'])
df = df.drop(columns=['baths'])
df = df.drop(columns=['D_mkt'])
df = df.drop(columns=['building_age'])
df = df.drop(columns=['lt'])
df = df.drop(columns=['lg'])
df = df.drop(columns=['ward'])

# data cleaning
df = df.dropna()
df['maint'] = df['maint'].fillna(df['maint'].median())
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# Define Feature Set (X) and Target (y)
X = df.drop(columns=['price'])
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost Approach
categorical_cols = [ 'DEN', 'neighbourhood', 'size']
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

one_hot = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot, categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

xgb_model = XGBRegressor(random_state=42)

xgb_param_grid = {
    'xgbregressor__n_estimators': [200, 500],
    'xgbregressor__max_depth': [4, 6, 8],
    'xgbregressor__learning_rate': [0.01, 0.1],
    'xgbregressor__reg_lambda': [1, 3, 5]
}

xgb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('xgbregressor', xgb_model)
])

xgb_gs = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_param_grid,
    scoring='neg_mean_squared_error',
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1,
    error_score='raise'
)

xgb_gs.fit(X_train, y_train)

final_pipeline = xgb_gs.best_estimator_

y_pred_xgb = final_pipeline.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)
mae = mean_absolute_error(y_test, y_pred_xgb)
mape = mean_absolute_percentage_error(y_test, y_pred_xgb)

print("===== XGBoost Results =====")
print("Best Params:", xgb_gs.best_params_)
print(f"Test RMSE: {xgb_rmse:,.0f}")
print(f"Test R^2:  {xgb_r2:.3f}")
print(f"MAE: {mae:,.0f}")
print(f"MAPE: {mape:.2%}")

# # Create a TreeExplainer for your XGBoost model.
# explainer = shap.TreeExplainer(final_pipeline.named_steps['xgbregressor'])

# # Transform your training data using the pipeline's preprocessor.
# # (This gives you the feature matrix as seen by the XGBoost model.)
# X_train_transformed = final_pipeline.named_steps['preprocessing'].transform(X_train)

# # Get the feature names from the preprocessor.
# # For scikit-learn version 1.0+ use get_feature_names_out
# feature_names = final_pipeline.named_steps['preprocessing'].get_feature_names_out()

# # Compute SHAP values.
# shap_values = explainer.shap_values(X_train_transformed)

# # Create a SHAP summary plot to visualize the global impact of features.
# shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names)


train_sizes, train_scores, val_scores = learning_curve(
    final_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
train_scores_mean = np.mean(-train_scores, axis=1)
val_scores_mean = np.mean(-val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.sqrt(train_scores_mean), 'o-', label="Training RMSE")
plt.plot(train_sizes, np.sqrt(val_scores_mean), 'o-', label="Validation RMSE")
plt.xlabel("Training Size")
plt.ylabel("RMSE")
plt.title("Learning Curve")
plt.legend()
plt.show()

errors = np.abs(y_pred_xgb - y_test)
pct_errors = errors / y_test

threshold = 0.2
within_threshold = pct_errors <= threshold
fraction_correct = np.mean(within_threshold)
print(f"{fraction_correct:.2%} of predictions are within ±20% of the actual price.")

thresholds = np.linspace(0, 0.3, 20)
accuracies = []

for t in thresholds:
    accuracies.append(np.mean(pct_errors <= t))
plt.figure(figsize=(8, 6))
plt.plot(thresholds * 100, np.array(accuracies) * 100, marker='o')
plt.xlabel("Error Threshold (%)")
plt.ylabel("Fraction of Predictions Within Threshold (%)")
plt.title("Accuracy vs. Error Threshold (0–20%)")
plt.show()

