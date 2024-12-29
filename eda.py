import pandas as pd
import matplotlib.pyplot as plt

ticker = ['MRNA','NVDA']
#df = pd.read_csv(f'consumerGoodsStocks/{ticker}_merged_data.csv')
df = pd.read_csv('all_stocks_merged.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')


ticker_data = df[df['Ticker'] == ticker]

daily_sentiment = ticker_data.groupby('Date')['Sentiment Score'].mean() 

daily_stock_price = ticker_data.groupby('Date')['Stock Price'].mean()


fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_title(f"Time Series of Stock Prices and Sentiment Trends for {ticker}", fontsize=16)
ax1.plot(daily_stock_price.index, daily_stock_price.values, label="Stock Price", color="blue", linewidth=2)
ax1.set_xlabel("Date", fontsize=14)
ax1.set_ylabel("Stock Price ($)", color="blue", fontsize=14)
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(daily_sentiment.index, daily_sentiment.values, label="Sentiment Score", color="orange", linewidth=2, linestyle="dashed")
ax2.set_ylabel("Sentiment Score", color="orange", fontsize=14)
ax2.tick_params(axis='y', labelcolor="orange")


fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=12)

correlation = daily_stock_price.corr(daily_sentiment)
print(f"Correlation coefficient: {correlation}")

plt.tight_layout()
plt.show()

