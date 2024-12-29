import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('all_stocks_merged.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

df = df.sort_values(by='Date')

def detect_outliers_z_score(data, threshold=3):
    #detect outliers using Z-score method.
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    return np.where(np.abs(z_scores) > threshold)

#outliers in sentiment scores
sentiment_outliers = detect_outliers_z_score(df['Sentiment Score'])
print(f"Sentiment Score Outliers: {df.iloc[sentiment_outliers]}")

plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Sentiment Score'], color='orange')
plt.title("Boxplot of Sentiment Scores (Outliers Highlighted)")
plt.show()

df['Price Change (%)'] = df['Stock Price'].pct_change() * 100

price_change_outliers = detect_outliers_z_score(df['Price Change (%)'].dropna())
print(f"Extreme Price Changes: {df.iloc[price_change_outliers]}")

plt.figure(figsize=(12, 6))
sns.histplot(df['Price Change (%)'], bins=50, kde=True, color='blue')
plt.axvline(df['Price Change (%)'].mean(), color='red', linestyle='dashed', label='Mean')
plt.axvline(df['Price Change (%)'].mean() + 3 * df['Price Change (%)'].std(), color='green', linestyle='dashed', label='+3 Std Dev')
plt.axvline(df['Price Change (%)'].mean() - 3 * df['Price Change (%)'].std(), color='green', linestyle='dashed', label='-3 Std Dev')
plt.legend()
plt.title("Distribution of Stock Price Changes with Outlier Thresholds")
plt.xlabel("Daily % Change in Stock Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Stock Price'], label='Stock Price', color='blue')
plt.scatter(df.iloc[price_change_outliers]['Date'], df.iloc[price_change_outliers]['Stock Price'], color='red', label='Outliers', zorder=5)
plt.title("Stock Price Time Series with Extreme Price Changes Highlighted")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
