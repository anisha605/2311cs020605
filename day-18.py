#1
import statistics
import numpy as np

stock_prices = [100, 102, 98, 105, 101, 97, 99, 103, 100, 98]

variance = statistics.variance(stock_prices)
print(f"Variance of stock prices: {variance}")

numpy_variance = np.var(stock_prices, ddof=1)
print(f"Variance of stock prices(using Numpy): {numpy_variance}")

#2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = {
    'Temperature': [23, 25, 38, 18, 21, 14, 14, 16, 29, 27, 28, 38, 31, 13, 14, 15, 33, 35, 37, 14, 24, 25, 28, 19, 18, 20],
    'Humidity': [65, 66, 77, 78, 70, 75, 98, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature'], df['Humidity'], color='blue', label='Temperature vs Humidity')

plt.title('Temperature vs Humidity', fontsize=14)
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Humidity', fontsize=12)
plt.legend()

plt.show()