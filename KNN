import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
rates = filteredData.loc[:, 'Spot Rate':'10Y Rate']

logReturns = np.log(rates / rates.shift(1)).dropna()

scaler = MinMaxScaler()
logReturnsNormalized = scaler.fit_transform(logReturns)

nearestNeighbour = 5
neighbors = NearestNeighbors(n_neighbors=nearestNeighbour, algorithm='auto', metric='euclidean')
neighbors.fit(logReturnsNormalized)

distances, indices = neighbors.kneighbors(logReturnsNormalized)
meanDistances = np.mean(distances, axis=1)
threshold = np.mean(meanDistances) + 3 * np.std(meanDistances)

anomalies = np.where(meanDistances > threshold)[0]

%matplotlib notebook
for i, column in enumerate(logReturns.columns):
    plt.figure(figsize=(12, 6))
    plt.plot(dates[1:], logReturns[column], label=f'Actual Log Returns {column}', color='navy')

    anomalyDates = dates[1:].iloc[anomalies]
    anomalyVals = logReturns[column].iloc[anomalies]
    plt.scatter(anomalyDates, anomalyVals, label='Anomalies', color='red', marker='o', s=50, edgecolors='black', linewidth=1.5)
    
    plt.title(f'Actual Log Returns {column} Over Time with Anomalies Detected using KNN')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid()
    plt.show()
    
    print("Number of anomalies:", len(anomalies))
