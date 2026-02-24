import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import PredictionErrorDisplay, r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt


from transformations import lag_all_features, add_lagged_targets,one_hot_encode_locations

INDEX_COLS = ['time_period', 'location']
FEATURE_COLUMNS = ["rainfall", "mean_temperature", "mean_relative_humidity"]
X_COLUMNS = INDEX_COLS + FEATURE_COLUMNS
# Load data and save target before transform (lagging removes the original column)
df = pd.read_csv("training_data.csv")
df = df.sort_values(["location", "time_period"]).copy()
disease_cases = df["disease_cases"].copy()

#df = lag_all_features(df[X_COLUMNS])
#print(df.columns)
# df = one_hot_encode_locations(df)
#print(df.columns)
# df = add_lagged_targets(df, disease_cases)
# print(df.columns)

# Re-attach target so NaN dropping stays in sync
df["disease_cases"] = disease_cases.values
df = df.dropna()

# Feature columns: everything except identifiers and target
exclude = ["time_period", "location", "disease_cases"]
X = df.drop(columns=exclude).select_dtypes(include="number")
y = df["disease_cases"].values

# Group by year so all months of a year stay together
groups = pd.to_datetime(df["time_period"]).dt.year.values

# Cross-validated predictions
cv = GroupKFold(n_splits=5)
model = GradientBoostingRegressor(random_state=42)
y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv)

# Metrics summary
print(f"R²:   {r2_score(y, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y, y_pred):.4f}")

# Visualize — both PredictionErrorDisplay modes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
PredictionErrorDisplay.from_predictions(y, y_pred, kind="actual_vs_predicted", ax=axes[0])
axes[0].set_title("Actual vs Predicted")
PredictionErrorDisplay.from_predictions(y, y_pred, kind="residual_vs_predicted", ax=axes[1])
axes[1].set_title("Residuals vs Predicted")
fig.suptitle("One-step GradientBoostingRegressor (GroupKFold by year)")
plt.tight_layout()
plt.show()
