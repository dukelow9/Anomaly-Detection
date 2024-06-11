import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#building custom ORC class
class OpticalReservoir(tf.keras.layers.Layer):
    def __init__(self, transmissionMatrix, activation=None, *args, **kwargs):
        self.transmissionMatrix = transmissionMatrix
        self._activation = tf.tanh if activation is None else activation
        super(OpticalReservoir, self).__init__(*args, **kwargs)

    def call(self, inputs):
        output = tf.matmul(inputs, self.transmissionMatrix[:, :inputs.shape[-1]])
        return self._activation(output)

#early stopping callback
earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)

#reading and filtering dataset
data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1_flagged.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
rates = filteredData.loc[:, 'Spot Rate':'10Y Rate']
anomalyFlag = filteredData['Anomaly Flag']

#log return calculation and normalisation
logReturns = np.log(rates / rates.shift(1)).dropna()
scaler = MinMaxScaler()
logReturnsNormalized = scaler.fit_transform(logReturns)

inputDim = logReturnsNormalized.shape[1]
activation = tf.keras.activations.relu
dim = inputDim

#building transmission matrix
transmissionMatrix = np.random.randn(dim, 100)/2 + 1j * np.random.randn(dim, 100)/2

opticalReservoir = OpticalReservoir(transmissionMatrix=transmissionMatrix, activation=activation)
opticalReservoir.trainable = False
output = tf.keras.layers.Dense(inputDim, kernel_regularizer=tf.keras.regularizers.l2(0.01), name="readouts")

optimizer = tf.keras.optimizers.Adam()

#calling the custom class to initialize the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(dim, activation='relu', input_shape=(None, inputDim)))
model.add(opticalReservoir)
model.add(output)
model.build(input_shape=(None, inputDim))
model.compile(loss="mse", optimizer=optimizer)
model.summary()

#training with recorded time
startTime = time.time()
history = model.fit(logReturnsNormalized, logReturnsNormalized, epochs=2000, batch_size=256,
                    shuffle=False, validation_split=0.2, callbacks=[earlyStopping])
endTime = time.time()

encodedLogReturnsNormalized = model.predict(logReturnsNormalized)
encodedLogReturns = scaler.inverse_transform(encodedLogReturnsNormalized)

#calculating errors and anomalies with set threshold
reconstructionErrors = np.mean(np.square(logReturnsNormalized - encodedLogReturnsNormalized), axis=1)
meanError = np.mean(reconstructionErrors)
threshold = meanError + 6 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

elapsedTime = endTime - startTime
print(f"Training Time: {elapsedTime} seconds")

#plotting actual and predicted data
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

#plotting loss curves
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

#plotting ROC curve
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
