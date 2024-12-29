from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

data = pd.read_csv('techStocks/TSLA_merged_data.csv')
target = 'TSLA'

encoder = OneHotEncoder(sparse_output=False)

encoded_features = encoder.fit_transform(data[['Sentiment Label', 'Topic']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

numerical_features = data[['Sentiment Score', 'Relevance Score', 'Date']]
X = pd.concat([numerical_features.reset_index(drop=True), encoded_df], axis=1)
y = data['Stock Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
}
gbr = GradientBoostingRegressor(random_state=42)
grid_search = RandomizedSearchCV(gbr, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42)
grid_search.fit(X_train, y_train)
best_gbr = grid_search.best_estimator_

xgb_model = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)

rf_model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=7)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)

best_gbr_pred = best_gbr.predict(X_test)
gbr_mse = mean_squared_error(y_test, best_gbr_pred)

results = {
    "Gradient Boosting (Best Params)": gbr_mse,
    "XGBoost": xgb_mse,
    "Random Forest": rf_mse,
}

if gbr_mse < xgb_mse and gbr_mse < rf_mse:
    model = best_gbr
    mse = gbr_mse
    feature_importances = model.feature_importances_
elif xgb_mse < rf_mse:
    model = xgb_model
    mse = xgb_mse
    feature_importances = model.feature_importances_
else:
    model = rf_model
    mse = rf_mse
    feature_importances = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance of Final Model')
plt.show()

print(results, mse)

