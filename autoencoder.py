import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
spotRates = filteredData['Spot Rate']

scaler = MinMaxScaler()
spotRates_normalized = scaler.fit_transform(np.array(spotRates).reshape(-1, 1))

inputLayer = Input(shape=(1,))
encoded = Dense(128, activation='relu')(inputLayer)
encoded = Dropout(0.2, name='dropout_layer')(encoded)
decoded = Dense(1, activation='linear')(encoded)

autoencoder = Model(inputLayer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
autoencoder.summary()
autoencoder.fit(spotRates_normalized, spotRates_normalized, epochs=150, batch_size=256, shuffle=False, validation_split=0.2)

encodedSpotRates_normalized = autoencoder.predict(spotRates_normalized)
encodedSpotRates = scaler.inverse_transform(encodedSpotRates_normalized)

reconstructionErrors = np.mean(np.square(spotRates_normalized - encodedSpotRates_normalized), axis=1)

mean_error = np.mean(reconstructionErrors)
threshold = mean_error + 2 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

plt.figure(figsize=(12, 6))
plt.plot(dates, spotRates, label='Actual Spot Rates', color='navy')
plt.plot(dates, encodedSpotRates, label='Predicted Spot Rates', color='yellowgreen', linewidth=0.7)

anomalyDates = dates.iloc[anomalies]
anomalyVals = pd.DataFrame(spotRates).values[anomalies, 0]
plt.scatter(anomalyDates, anomalyVals, label='Anomalies', color='red', marker='o')

plt.xlabel('Date')
plt.ylabel('Spot Rate')
plt.legend()
plt.title(f'Spot Rate Anomaly Detection for {currency}')
plt.grid()
plt.show()

print("Number of anomalies:", len(anomalies))
