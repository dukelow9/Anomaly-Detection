import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def sampling(args):
    zMean, zLogVar = args
    batch = tf.shape(zMean)[0]
    dim = tf.shape(zMean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return zMean + tf.exp(0.5 * zLogVar) * epsilon

class CustomVariationalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.zMean = None
        self.zLogVar = None

    def vae_loss(self, x, xDecodedMean):
        xentLoss = tf.reduce_mean(tf.square(x - xDecodedMean))
        klLoss = -0.5 * tf.reduce_mean(1 + self.zLogVar - tf.square(self.zMean) - tf.exp(self.zLogVar))
        return xentLoss + klLoss

    def call(self, inputs):
        x, zMean, zLogVar = inputs
        self.zMean = zMean
        self.zLogVar = zLogVar
        decoded_output = x
        loss = self.vae_loss(x, decoded_output)
        self.add_loss(loss, inputs=inputs)
        return x

    def compute_output_shape(self, inputShape):
        return inputShape

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
spotRates = filteredData['Spot Rate']

scaler = MinMaxScaler()
spotRatesNormalized = scaler.fit_transform(np.array(spotRates).reshape(-1, 1))

inputLayer = Input(shape=(1,))
encoded = Dense(128, activation='relu')(inputLayer)
encoded = Dropout(0.2)(encoded)

zMean = Dense(2)(encoded)
zLogVar = Dense(2)(encoded)
z = Lambda(sampling, output_shape=(2,))([zMean, zLogVar])

decoded = Dense(1, activation='linear')(z)

lossLayer = CustomVariationalLayer()([inputLayer, zMean, zLogVar])

vae = Model(inputLayer, lossLayer)
vae.compile(optimizer=Adam(learning_rate=0.001), loss=None)
vae.summary()
startTime = time.time()
history = vae.fit(spotRatesNormalized, epochs=150, batch_size=256, shuffle=False, validation_data=(spotRatesNormalized, None))
endTime = time.time()

encodedSpotRatesNormalized = vae.predict(spotRatesNormalized)
encodedSpotRates = scaler.inverse_transform(encodedSpotRatesNormalized)

reconstructionErrors = np.mean(np.square(spotRatesNormalized - encodedSpotRatesNormalized), axis=1)

meanError = np.mean(reconstructionErrors)
threshold = meanError + 3 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

elapsedTime = endTime - startTime
print(f"Training Time: {elapsedTime} seconds")

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
