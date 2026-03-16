# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
stock_data = pd.read_csv(r'C:\Users\Welcome\Desktop\Readme_Sample\dataset\google.csv')
stock_data.head()
stock_data.shape
stock_data.columns

# Convert Datetime from object 
stock_data['Date']= pd.to_datetime(stock_data['Date'] ,format('%d/%m/%Y'))
stock_data.info()

# Plot graph for stock price
plt.figure(figsize=(12,5))
plt.plot(stock_data['Close'])
plt.title("Google Stock Price Trend ")
plt.show()