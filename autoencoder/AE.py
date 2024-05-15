import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1_flagged.xlsx")

currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
rates = filteredData.loc[:, 'Spot Rate':'10Y Rate']
anomalyFlag = filteredData['Anomaly Flag']

logReturns = np.log(rates / rates.shift(1)).dropna()

scaler = MinMaxScaler()
logReturnsNormalized = scaler.fit_transform(logReturns)

inputDim = logReturnsNormalized.shape[1]
inputLayer = Input(shape=(inputDim,))
dropoutLayer = Dropout(0.2, name='dropout_layer')(inputLayer)
encoded = Dense(8, activation='relu')(dropoutLayer)
encoded = Dense(4, activation='relu')(encoded)
encoded = Dense(2, activation='relu')(encoded)
decoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(decoded)
decoded = Dense(inputDim, activation='linear')(decoded)

autoencoder = Model(inputLayer, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')
autoencoder.summary()

earlyStop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

startTime = time.time()

history = autoencoder.fit(logReturnsNormalized, logReturnsNormalized, epochs=1000, batch_size=256, shuffle=False,
                          validation_split=0.2, callbacks=earlyStop)
endTime = time.time()

encodedLogReturnsNormalized = autoencoder.predict(logReturnsNormalized)
encodedLogReturns = scaler.inverse_transform(encodedLogReturnsNormalized)

reconstructionErrors = np.mean(np.square(logReturnsNormalized - encodedLogReturnsNormalized), axis=1)

meanError = np.mean(reconstructionErrors)
threshold = meanError + 6 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

elapsedTime = endTime - startTime
print(f"Training Time: {elapsedTime} seconds")

minLength = min(len(dates[1:]), len(logReturns), len(encodedLogReturns))

%matplotlib notebook
for i, column in enumerate(logReturns.columns):
    plt.figure(figsize=(12, 6))
    plt.plot(dates[1: minLength + 1], logReturns[column][:minLength], label=f'Actual Log Returns {column}', color='navy')
    plt.plot(dates[1: minLength + 1], encodedLogReturns[:minLength, i], label=f'Predicted Log Returns {column}', color='yellowgreen',
             linewidth=1, linestyle='--')

    anomalyDates = dates[1: minLength + 1].iloc[anomalies]
    anomalyVals = encodedLogReturns[:minLength, i][anomalies]
    plt.scatter(anomalyDates, anomalyVals, label='Anomalies', color='red', marker='o', s=50, edgecolors='black',
                linewidth=1.5)
    
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

from sklearn.metrics import roc_auc_score, roc_curve

minLength = min(len(dates[1:]), len(logReturns), len(encodedLogReturns))
trueLabels = anomalyFlag[1: minLength + 1]
ROCAUC = roc_auc_score(trueLabels, reconstructionErrors)
fpr, tpr, thresholds = roc_curve(trueLabels, reconstructionErrors)

plt.figure()
plt.plot(fpr, tpr, color='lime', lw=2, label='AUC = {:.2f}'.format(ROCAUC))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
