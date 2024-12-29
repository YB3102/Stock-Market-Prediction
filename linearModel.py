import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

class StockPredictor:
    def __init__(self, file_path):
        
        self.df = pd.read_csv(file_path)
        
        self.preprocess_data()
        
        self.prepare_features()
    
    def preprocess_data(self):
    
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y%m%d')
        
        self.engineer_features()

    #features for prediction
    def engineer_features(self):

        #sentiment label to numeric
        sentiment_order = ['Bearish', 'Somewhat-Bearish', 'Neutral', 'Somewhat_Bullish', 'Bullish']
        self.df['sentiment_numeric'] = pd.Categorical(
            self.df['Sentiment Label'], 
            categories=sentiment_order,
            ordered=True
        ).codes
        
        #time-based features
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
        
        #rolling stats
        self.df['sentiment_rolling_mean'] = (
            self.df.groupby('Ticker')['Sentiment Score'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )
        self.df.dropna(inplace=True)
    
    #prepare features and target variable
    def prepare_features(self):
        
        #select features for prediction
        self.features = [
            'Sentiment Score', 'Relevance Score', 'sentiment_numeric', 
            'month', 'day_of_week', 'prev_sentiment_score', 
            'prev_stock_price', 'sentiment_rolling_mean',
            'Sentiment Label', 'Topic'
        ]
        
        #separate features and target
        self.X = self.df[self.features]
        self.y = self.df['Stock Price']
    
    def create_preprocessing_pipeline(self):
        #returns:
        # ColumnTransformer: preprocessor for features

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
    
    #creates egression models
    def create_advanced_models(self):  

        #returns:
        # dict: dictionary of models with their parameter grids
        
        models = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'solver': ['auto', 'svd', 'cholesky']
                }
            }
        }
        
        return models
    
    def perform_grid_search(self, model, params):
        #parameters:
        # model (sklearn estimator): Model to tune
        # params (dict): Hyperparameter grid
    
        #returns:
        # GridSearchCV: best model from grid search

        preprocessor = self.create_preprocessing_pipeline()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline, 
            param_grid={f'regressor__{k}': v for k, v in params.items()}, 
            cv=5, 
            scoring='neg_mean_squared_error'
        )
       
        grid_search.fit(self.X, self.y)
        
        return grid_search
    
    def create_ensemble_model(self, best_models):        
        #parameters:
        #best_models:list of best performing models
        
        #returns:
        # VotingRegressor: ensemble of models

        #create ensemble of best models
        ensemble = VotingRegressor(
            estimators=[('model', model) for model in best_models]
        )
        
        return ensemble
    
    def train_and_evaluate(self):
        #split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
       
        models = self.create_advanced_models()
        
        best_models = []
        best_performance = float('inf')
        best_model_info = None
        
        for name, model_info in models.items():
            grid_search = self.perform_grid_search(
                model_info['model'], 
                model_info['params']
            )
            
            #evaluate best model
            y_pred = grid_search.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name.upper()} Model Performance:")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R-squared Score: {r2:.4f}\n")
            
            #plot predictions
            self.plot_predictions(y_test, y_pred, title="Model Predictions")
            
            if mse < best_performance:
                best_performance = mse
                best_model_info = {
                    'name': name,
                    'model': grid_search.best_estimator_,
                    'mse': mse,
                    'r2': r2
                }
            best_models.append(grid_search.best_estimator_)
        
        ensemble_model = self.create_ensemble_model(best_models)
       
        ensemble_model.fit(X_train, y_train)
        y_pred_ensemble = ensemble_model.predict(X_test)
        ensemble_mse = mean_squared_error(y_test, y_pred_ensemble)
        ensemble_r2 = r2_score(y_test, y_pred_ensemble)
        
        print("ENSEMBLE Model Performance:")
        print(f"Mean Squared Error: {ensemble_mse:.4f}")
        print(f"R-squared Score: {ensemble_r2:.4f}\n")
        
        self.plot_predictions(y_test, y_pred_ensemble, title="ENSEMBLE Model Predictions")
        
        return {
            'best_model': best_model_info,
            'ensemble_performance': {
                'mse': ensemble_mse,
                'r2': ensemble_r2
            }
        }
    def predict_future(self, start_date, days_to_predict, initial_features, output_csv_path):
        # Predict future stock prices for a given number of days and save to CSV.
        # Parameters:
        # -start_date (datetime): The date to start predictions from.
        # -days_to_predict (int): Number of days to predict into the future.
        # -initial_features (dict): Initial feature values for the starting date.
        # -output_csv_path (str): Path to save the CSV file with predictions.
        # Returns:
        # - DataFrame with predicted stock prices and features for each future day.

        future_predictions = []
        current_features = initial_features.copy()

        for day in range(days_to_predict):
            input_data = pd.DataFrame([current_features])

            preprocessor = self.create_preprocessing_pipeline()
            input_data_transformed = preprocessor.transform(input_data)

            predicted_price = self.best_model.predict(input_data_transformed)[0]

            current_date = start_date + pd.Timedelta(days=day)
            future_predictions.append({
                'Date': current_date,
                'Predicted Stock Price': predicted_price,
            })

            current_features['prev_stock_price'] = predicted_price
            current_features['Date'] = current_date
            current_features['day_of_week'] = current_date.weekday()
            current_features['month'] = current_date.month

        future_predictions_df = pd.DataFrame(future_predictions)

        future_predictions_df.to_csv(output_csv_path, index=False)

        print(f"Predictions saved to {output_csv_path}")

        return future_predictions_df

    def plot_predictions(self, y_test, y_pred, title):
        
        #Visualize actual vs predicted stock prices
        # Parameters:
        # y_test: Actual stock prices
        # y_pred: Predicted stock prices

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 color='red', linestyle='--', lw=2)
        plt.title(title)
        plt.xlabel('Actual Stock Price')
        plt.ylabel('Predicted Stock Price')
        plt.tight_layout()
        plt.show()

def main(file_path):
    # Returns:
    # dict: Model performance metrics
 
    # Create predictor
    predictor = StockPredictor(file_path)
    
    # Train and evaluate models
    performance = predictor.train_and_evaluate()
    
    return performance


if __name__ == '__main__':

    file_path = 'all_stocks_merged.csv'

    results = main(file_path)