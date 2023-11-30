import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'JPY'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
spotRates = filteredData['Spot Rate']

scaler = MinMaxScaler()
spotRatesNormalized = scaler.fit_transform(np.array(spotRates).reshape(-1, 1))

inputLayer = Input(shape=(1,))
encoded = Dense(128, activation='relu')(inputLayer)
encoded = Dropout(0.2, name='dropout_layer')(encoded)
decoded = Dense(1, activation='linear')(encoded)

autoencoder = Model(inputLayer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
autoencoder.summary()
start_time = time.time()
history = autoencoder.fit(spotRatesNormalized, spotRatesNormalized, epochs=150, batch_size=256, shuffle=False, validation_split=0.2)
end_time = time.time()

encodedspotRatesNormalized = autoencoder.predict(spotRatesNormalized)
encodedSpotRates = scaler.inverse_transform(encodedspotRatesNormalized)

reconstructionErrors = np.mean(np.square(spotRatesNormalized - encodedspotRatesNormalized), axis=1)

meanError = np.mean(reconstructionErrors)
threshold = meanError + 4 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time} seconds")

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


plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='mediumpurple')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

last_loss = history.history['loss'][-1]
print(f"Last training loss: {last_loss}")
