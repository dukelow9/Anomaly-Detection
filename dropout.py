import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
spotRates = filteredData['Spot Rate']

maxSpotRate = max(spotRates)
spotRates = spotRates / maxSpotRate

inputLayer = Input(shape=(1,))
encoded = Dense(128, activation='relu')(inputLayer)
decoded = Dense(1, activation='linear')(encoded)

autoencoder = Model(inputLayer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
autoencoder.summary()
autoencoder.fit(spotRates, spotRates, epochs=150, batch_size=256, shuffle=False, validation_split=0.2)
encodedSpotRates = autoencoder.predict(spotRates)

reconstructionErrors = []
for i in range(len(spotRates)):
    error = np.mean(np.square(spotRates.iloc[i] - encodedSpotRates[i].flatten()))
    reconstructionErrors.append(error)

mean_error = np.mean(reconstructionErrors)
threshold = np.percentile(reconstructionErrors, 99)
anomalies = np.atleast_1d(np.array(reconstructionErrors) > threshold).nonzero()
