import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1_AED.xlsx")
dates = data['Date']
spotRates = data['Spot Rate']

maxSpotRate = max(spotRates)
spotRates = spotRates / maxSpotRate

inputLayer = Input(shape=(1,))
encoded = Dense(128, activation='relu')(inputLayer)
decoded = Dense(1, activation='linear')(encoded)

autoencoder = Model(inputLayer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
autoencoder.fit(spotRates, spotRates, epochs=50, batch_size=256, shuffle=False, validation_split=0.2)
encodedSpotRates = autoencoder.predict(spotRates)

reconstructionErrors = []
for i in range(len(spotRates)):
    error = np.mean(np.square(spotRates[i] - encodedSpotRates[i].flatten()))
    reconstructionErrors.append(error)

threshold = 1e-9
anomalies = np.atleast_1d(np.array(reconstructionErrors) > threshold).nonzero()

%matplotlib notebook
plt.figure(figsize=(12, 6))
plt.plot(dates, spot_rates * maxSpotRate, label='Actual Spot Rates', color='navy')
plt.plot(dates, encodedSpotRates * maxSpotRate, label='Predicted Spot Rates', color='yellowgreen')

anomalyIndices = anomalies[0]
anomalyDates = [dates[i] for i in anomalyIndices]
anomalyVals = [spot_rates[i] * maxSpotRate for i in anomalyIndices]
plt.scatter(anomalyDates, anomalyVals, label='Anomalies', color='red', marker='o')

plt.xlabel('Date')
plt.ylabel('Spot Rate')
plt.legend()
plt.title('AED Spot Rates')
plt.grid()
plt.show()

print("Number of anomalies:", len(anomalies[0]))
