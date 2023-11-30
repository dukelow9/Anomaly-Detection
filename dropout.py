import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1_AED.xlsx")
dates = data['Date']
spotRates = data['Spot Rate']

scaler = MinMaxScaler()
spotRates = scaler.fit_transform(np.array(spotRates).reshape(-1, 1))

inputLayer = Input(shape=(1,))
encoded = Dense(128, activation='relu')(inputLayer)
encoded = Dropout(0.2, name='dropout_layer')(encoded)
decoded = Dense(1, activation='linear')(encoded)

autoencoder = Model(inputLayer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
autoencoder.summary()
autoencoder.fit(spotRates, spotRates, epochs=150, batch_size=256, shuffle=False, validation_split=0.2)
encodedSpotRates = autoencoder.predict(spotRates)
encodedSpotRates = scaler.inverse_transform(encodedSpotRates)

reconstructionErrors = np.mean(np.square(spotRates - encodedSpotRates), axis=1)

mean_error = np.mean(reconstructionErrors)
threshold = np.percentile(reconstructionErrors, 99)
anomalies = np.where(reconstructionErrors > threshold)[0]
