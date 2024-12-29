import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.read_csv('techStocks/TSLA_merged_data.csv')

target = 'TSLA'
features = ['Sentiment Score', 'Sentiment Label', 'Topic', 'Relevance Score']

encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[['Sentiment Label', 'Topic']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

numerical_features = data[['Sentiment Score', 'Relevance Score']]
X = pd.concat([numerical_features.reset_index(drop=True), encoded_df], axis=1)
y = data['Stock Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False, inplace=True)
print("Feature Importances:")
print(feature_importances)
