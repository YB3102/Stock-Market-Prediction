# Stock Price Prediction Using Sentiment Analysis
## Project Overview

This project explores the relationship between news sentiment and stock price movements while building predictive models for stock prices. It combines Exploratory Data Analysis (EDA), feature engineering, and machine learning techniques (Random Forest, Ridge Regression, and Ensemble Learning) to understand and predict stock price trends.
Goals

    Analyze how sentiment data and other features impact stock price fluctuations.
    Train and evaluate multiple regression models to predict future stock prices.
    Investigate the role of sentiment-driven features, relevance scores, and rolling statistics in model performance.
    Utilize ensemble learning for robust stock price predictions.

## Dataset Description

The input dataset (all_stocks_merged.csv) consists of:

    Date: Date of the news/article.
    Ticker: Stock symbol 
    Sentiment Score: Numerical sentiment score between -1 (negative) and +1 (positive).
    Sentiment Label: Categorical sentiment labels (e.g., Bearish, Neutral, Bullish).
    Topic: News topic (e.g., "Earnings", "Technology").
    Relevance Score: Importance of the news/article (0 to 1).
    Stock Price: Actual stock closing price.

## Project Components
1. Exploratory Data Analysis (EDA)

The EDA focuses on:

    Sentiment Score Distributions: Understanding the spread and average sentiment values.
    Time-Based Analysis: How stock prices and sentiment vary over months, weekdays, and specific periods (e.g., presidential election seasons).
    Correlation Analysis: Sentiment scores, relevance, and their impact on stock prices.
    Rolling Features: Moving averages of sentiment scores to capture trends over short windows.
    Outlier Detection: Identifying significant price changes or anomalies during events.

2. Feature Engineering

The following features were engineered to improve model accuracy:

    Numeric Sentiment: Mapping sentiment labels (e.g., Bearish → 0, Bullish → 4).
    Time-Based Features: Month, day of the week.
    Lagged Features: Previous day's sentiment score and stock price.
    Rolling Statistics: 3-day moving average of sentiment scores for each stock ticker.

3. Machine Learning Pipeline

The pipeline builds advanced models using scikit-learn's regression techniques.
Pipeline Components

    Preprocessing:
        Standardization for numeric features using StandardScaler.
        Polynomial feature expansion for numeric data.
        One-hot encoding for categorical features (e.g., Sentiment Labels, Topics).
    Models:
        Random Forest Regressor: Captures non-linear relationships.
        Ridge Regression: Linear regression with L2 regularization.
        Voting Ensemble: Combines multiple models to improve prediction robustness.

Model Tuning

    Grid Search with Cross-Validation (GridSearchCV) is applied to optimize hyperparameters for both models.

4. Ensemble Learning

The Voting Regressor is used to combine predictions from the best-performing models:

    Aggregates predictions from Random Forest and Ridge Regression.
    Ensures stability and reduces variance.

Code Explanation
StockPredictor Class

The StockPredictor class handles all key project tasks, including:

    Data Loading and Preprocessing:
        Converts dates to datetime format.
        Maps sentiment labels to numeric codes.
        Adds lagged features and rolling averages.
    Feature Preparation:
        Separates predictor variables (X) and target variable (y).
    Preprocessing Pipeline:
        Numeric features: Scaled and expanded using Polynomial Features.
        Categorical features: Encoded using One-Hot Encoding.
    Model Training and Evaluation:
        Implements Grid Search Cross-Validation for hyperparameter tuning.
        Evaluates performance using Mean Squared Error (MSE) and R² Score.
    Visualization:
        Actual vs. Predicted stock price scatter plots for each model.
    Ensemble Model:
        Combines the best models for enhanced prediction accuracy.

## Predicting Future Prices

The predict_future function generates future predictions by iteratively forecasting stock prices over a given time horizon. The output is saved as a CSV file.
## How to Run the Project
1. Install Dependencies

Ensure you have the required libraries:

pip install pandas numpy matplotlib scikit-learn

2. Project Execution

Run the following script from: eda.py, eda2.py

python script_to_run.py

3. Example Workflow

    Train and Evaluate Models: The train_and_evaluate method will:
        Train Random Forest and Ridge models.
        Tune hyperparameters using Grid Search.
        Combine predictions using the ensemble model.
    Visualize Predictions: Scatter plots of actual vs. predicted prices will be displayed for each model.
    Predict Future Prices: Use the predict_future method to forecast future stock prices.

Sample Output

The script outputs:

    Best Model Performance:

RIDGE Model Performance:
Best Parameters: {'alpha': 1.0, 'solver': 'auto'}
Mean Squared Error: 2.3456
R-squared Score: 0.8901

Ensemble Performance:

    ENSEMBLE Model Performance:
    Mean Squared Error: 2.1004
    R-squared Score: 0.9023

    Scatter Plots: Visualizations of actual vs. predicted stock prices.

