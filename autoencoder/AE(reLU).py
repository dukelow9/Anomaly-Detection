import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
rates = filteredData.loc[:, 'Spot Rate':'10Y Rate']

logReturns = np.log(rates / rates.shift(1)).dropna()

scaler = MinMaxScaler()
logReturnsNormalized = scaler.fit_transform(logReturns)

inputDim = logReturnsNormalized.shape[1]
encodingDim = 4
inputLayer = Input(shape=(inputDim,))
encoded = Dense(encodingDim, activation='relu')(inputLayer)
encoded = Dropout(0.02, name='dropout_layer')(encoded)
decoded = Dense(inputDim, activation='sigmoid')(encoded)

autoencoder = Model(inputLayer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.summary()

earlyStop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

startTime = time.time()
history = autoencoder.fit(logReturnsNormalized, logReturnsNormalized, epochs=1000, batch_size=256, shuffle=False,
                          validation_split=0.2, callbacks=earlyStop)
endTIme = time.time()

encodedLogReturnsNormalized = autoencoder.predict(logReturnsNormalized)
encodedLogReturns = scaler.inverse_transform(encodedLogReturnsNormalized)

reconstructionErrors = np.mean(np.square(logReturnsNormalized - encodedLogReturnsNormalized), axis=1)

mean_error = np.mean(reconstructionErrors)
threshold = mean_error + 2 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

elapsedTime = endTIme - startTime
print(f"Training Time: {elapsedTime} seconds")

%matplotlib notebook
for i, column in enumerate(logReturns.columns):
    plt.figure(figsize=(12, 6))
    plt.plot(dates[1:], logReturns[column], label=f'Actual Log Returns {column}', color='navy')
    plt.plot(dates[1:], encodedLogReturns[:, i], label=f'Predicted Log Returns {column}', color='yellowgreen', linewidth=1, linestyle='--')

    anomalyDates = dates[1:].iloc[anomalies]
    anomalyVals = encodedLogReturns[:, i][anomalies]
    plt.scatter(anomalyDates, anomalyVals, label='Anomalies', color='red', marker='o', s=50, edgecolors='black', linewidth=1.5)
    
    plt.title(f'Actual vs Predicted Log Returns {column} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
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

lastLoss = history.history['loss'][-1]
print(f"Last training loss: {lastLoss}")
