import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('all_stocks_merged.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df = df.sort_values(by='Date')

#sentiment Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Sentiment Score'], kde=True, bins=50, color='blue')
plt.axvline(-0.35, color='red', linestyle='--', label='Bearish Threshold')
plt.axvline(-0.15, color='orange', linestyle='--', label='Somewhat Bearish Threshold')
plt.axvline(0.15, color='green', linestyle='--', label='Somewhat Bullish Threshold')
plt.axvline(0.35, color='purple', linestyle='--', label='Bullish Threshold')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Score Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.show()