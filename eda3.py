import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('all_stocks_merged.csv')


df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df = df.sort_values(by='Date')

df['Price Change (%)'] = df['Stock Price'].pct_change() * 100

df = df.dropna()

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df['Normalized Sentiment Score'] = normalize(df['Sentiment Score'])
df['Normalized Price Change'] = normalize(df['Price Change (%)'])

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Normalized Sentiment Score'], label='Normalized Sentiment Score', color='orange', linewidth=2)
plt.plot(df['Date'], df['Normalized Price Change'], label='Normalized Price Change (%)', color='blue', linestyle='dashed', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Normalized Values')
plt.title('Cross-Relationship Visualization: Sentiment Score vs. Price Change')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Sentiment Score'], y=df['Price Change (%)'], hue=df['Sentiment Label'], palette='coolwarm', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='No Price Change')
plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Neutral Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Price Change (%)')
plt.title('Scatter Plot: Sentiment Score vs. Price Change')
plt.legend(title='Sentiment Label')
plt.grid(alpha=0.3)
plt.show()
