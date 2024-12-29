import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    explained_variance_score
)

class RidgeStockPredictor:
    def __init__(self, file_path):
    
        self.df = pd.read_csv(file_path)

        self.preprocess_data()

        self.prepare_features()
    
    def preprocess_data(self):

        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y%m%d')
  
        self.engineer_features()
    
    def engineer_features(self):

        sentiment_order = ['Bearish', 'Somewhat-Bearish', 'Neutral', 'Somewhat_Bullish', 'Bullish']
        self.df['sentiment_numeric'] = pd.Categorical(
            self.df['Sentiment Label'], 
            categories=sentiment_order,
            ordered=True
        ).codes
  
        self.df['month'] = self.df['Date'].dt.month
        self.df['day_of_week'] = self.df['Date'].dt.dayofweek

        for ticker in self.df['Ticker'].unique():
            ticker_mask = self.df['Ticker'] == ticker

            self.df.loc[ticker_mask, 'prev_sentiment_score'] = (
                self.df.loc[ticker_mask, 'Sentiment Score'].shift(1)
            )
            
            self.df.loc[ticker_mask, 'prev_stock_price'] = (
                self.df.loc[ticker_mask, 'Stock Price'].shift(1)
            )

        self.df['sentiment_rolling_mean'] = (
            self.df.groupby('Ticker')['Sentiment Score'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )
        
        self.df.dropna(inplace=True)
    
    def prepare_features(self):
    
        self.features = [
            'Sentiment Score', 'Relevance Score', 'sentiment_numeric', 
            'month', 'day_of_week', 'prev_sentiment_score', 
            'prev_stock_price', 'sentiment_rolling_mean',
            'Sentiment Label', 'Topic'
        ]
      
        self.X = self.df[self.features]
        self.y = self.df['Stock Price']
    
    def create_preprocessing_pipeline(self):

        numeric_features = [
            'Sentiment Score', 'Relevance Score', 'sentiment_numeric', 
            'month', 'day_of_week', 'prev_sentiment_score', 
            'prev_stock_price', 'sentiment_rolling_mean'
        ]
        
        categorical_features = ['Sentiment Label', 'Topic']
        
        preprocessor = ColumnTransformer(
            transformers=[
               
                ('num', Pipeline([
                    ('scaler', StandardScaler()),
                    ('poly', PolynomialFeatures(degree=2, include_bias=False))
                ]), numeric_features),
                
                ('cat', OneHotEncoder(
                    handle_unknown='ignore', 
                    max_categories=5
                ), categorical_features)
            ])
        
        return preprocessor
    
    def train_and_evaluate_ridge(self):
      
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        preprocessor = self.create_preprocessing_pipeline()
        
        ridge_params = {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
       
        ridge_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(random_state=42))
        ])
       
        grid_search = GridSearchCV(
            estimator=ridge_pipeline, 
            param_grid={f'regressor__{k}': v for k, v in ridge_params.items()}, 
            cv=5, 
            scoring='neg_mean_squared_error'
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
       
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        
        print("Ridge Regression Model Performance:")
        print("\n--- Model Configuration ---")
        print(f"Best Parameters: {grid_search.best_params_}")
        
        print("\n--- Performance Metrics ---")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²) Score: {r2:.4f}")
        print(f"Explained Variance Score: {explained_variance:.4f}")
      
        feature_names = (
            preprocessor.named_transformers_['num'].named_steps['poly'].get_feature_names_out(
                numeric_features
            ).tolist() + 
            preprocessor.named_transformers_['cat'].get_feature_names_out(
                categorical_features
            ).tolist()
        )
        
        coefficients = best_model.named_steps['regressor'].coef_
        
        print("\n--- Feature Importances ---")
        feature_importance = sorted(
            zip(feature_names, coefficients), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        for feature, importance in feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")
      
        self.plot_predictions(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_variance,
            'best_params': grid_search.best_params_
        }
    
    def plot_predictions(self, y_test, y_pred):
       
        plt.figure(figsize=(12, 7))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 color='red', linestyle='--', lw=2)
        plt.title('Ridge Regression: Actual vs Predicted Stock Prices')
        plt.xlabel('Actual Stock Price')
        plt.ylabel('Predicted Stock Price')
        plt.tight_layout()
        plt.show()


def main(file_path):
    
    predictor = RidgeStockPredictor(file_path)
    
    # train and evaluate ridge model
    performance = predictor.train_and_evaluate_ridge()
    
    return performance

if __name__ == '__main__':

    file_path = 'consumerGoodsStocks/consumerGoodsStocks_merged_data.csv'
   
    results = main(file_path)