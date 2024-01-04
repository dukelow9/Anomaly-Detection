import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras. callbacks import EarlyStopping

class RNNCell(tf.keras.layers.Layer):
    def __init__(self, neurons, decay=0.1, alpha=0.5, spectralRad=0.9, scale=1.0, seed=None, epsilon=None, sparseness=0.0,
                 activation=None, optimize=False, optimizeVars=None, *args, **kwargs):
        self.seed = seed
        self.neurons = neurons
        self.state_size = neurons
        self.sparseness = sparseness
        self.decay = decay
        self.alpha = alpha
        self.spectralRad = spectralRad
        self.scale = scale
        self.epsilon = epsilon
        self._activation = tf.tanh if activation is None else activation
        self.optimize = optimize
        self.optimizeVars = optimizeVars

        super(RNNCell, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.optimize_table = {var: var in self.optimizeVars for var in ["alpha", "spectralRad", "decay", "scale"]}
        self.decay = self.add_weight(initializer=initializers.constant(self.decay),
                                     trainable=self.optimize_table["decay"], dtype=tf.float32)
        self.alpha = self.add_weight(initializer=initializers.constant(self.alpha),
                                     trainable=self.optimize_table["alpha"], dtype=tf.float32)
        self.spectralRad = self.add_weight(initializer=initializers.constant(self.spectralRad),
                                   trainable=self.optimize_table["spectralRad"], dtype=tf.float32)
        self.scale = self.add_weight(initializer=initializers.constant(self.scale),
                                  trainable=self.optimize_table["scale"], dtype=tf.float32)
        self.storeAlpha = self.add_weight(initializer=initializers.constant(self.alpha),
                                           trainable=False, dtype=tf.float32)
        self.ratio = self.add_weight(initializer=initializers.constant(1.0),
                                          trainable=False, dtype=tf.float32)
        self.kernel = self.add_weight(shape=(input_shape[-1], self.neurons), 
                                      initializer=initializers.RandomUniform(-1, 1, seed=self.seed), trainable=False)
        self.recurrent_kernel_init = self.add_weight(shape=(self.neurons, self.neurons),
                                                     initializer=initializers.RandomNormal(seed=self.seed), trainable=False)
        self.recurrent_kernel = self.add_weight(shape=(self.neurons, self.neurons), initializer=tf.zeros_initializer(),
                                                trainable=False)
        self.recurrent_kernel_init.assign(self.setSparseness(self.recurrent_kernel_init))
        self.recurrent_kernel.assign(self.setAlpha(self.recurrent_kernel_init))
        self.ratio.assign(self.echoStateRatio(self.recurrent_kernel))
        self.spectralRad.assign(self.findEchoStatespectralRad(self.recurrent_kernel * self.ratio))

        self.built = True

    def setAlpha(self, W):
        return 0.5 * (self.alpha * (W + tf.transpose(W)) + (1 - self.alpha) * (W - tf.transpose(W)))

    def setSparseness(self, W):
        mask = tf.cast(tf.random.uniform(W.shape, seed=self.seed) < (1 - self.sparseness), dtype=W.dtype)
        return W * mask

    def echoStateRatio(self, W):
        eigvals = tf.linalg.eigvals(W)
        return tf.reduce_max(tf.abs(eigvals))

    def findEchoStatespectralRad(self, W):
        target = 1.0
        eigvals = tf.linalg.eigvals(W)
        x = tf.math.real(eigvals)
        y = tf.math.imag(eigvals)
        a = x**2 * self.decay**2 + y**2 * self.decay**2
        b = 2 * x * self.decay - 2 * x * self.decay**2
        c = 1 + self.decay**2 - 2 * self.decay - target**2
        sol = (tf.sqrt(b**2 - 4*a*c) - b)/(2*a)
        spectralRad = tf.reduce_min(sol)
        return spectralRad

    def call(self, inputs, states):
        rkernel = self.setAlpha(self.recurrent_kernel_init)
        if self.alpha != self.storeAlpha:
            self.decay.assign(tf.clip_by_value(self.decay, 0.00000001, 0.25))
            self.alpha.assign(tf.clip_by_value(self.alpha, 0.000001, 0.9999999))
            self.spectralRad.assign(tf.clip_by_value(self.spectralRad, 0.5, 1.0e100))
            self.scale.assign(tf.clip_by_value(self.scale, 0.5, 1.0e100))

            self.ratio.assign(self.echoStateRatio(rkernel))
            self.spectralRad.assign(self.findEchoStatespectralRad(rkernel * self.ratio))
            self.storeAlpha.assign(self.alpha)

        ratio = self.spectralRad * self.ratio * (1 - self.epsilon)
        previousOutput = states[0]
        output = previousOutput + self.decay * (
                tf.matmul(inputs, self.kernel * self.scale) +
                tf.matmul(self._activation(previousOutput), rkernel * ratio)
                - previousOutput)

        return self._activation(output), [output]
    
earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)
    
###########################################################################################################

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
rates = filteredData.loc[:, 'Spot Rate':'10Y Rate']

logReturns = np.log(rates / rates.shift(1)).dropna()

scaler = MinMaxScaler()
logReturnsNormalized = scaler.fit_transform(logReturns)

inputDim = logReturnsNormalized.shape[1]

cell = RNNCell(neurons=100, activation=tf.keras.activations.relu, decay=0.02, alpha=0.5, spectralRad=0.9, scale=1.0, epsilon=None,
                        sparseness=0.0, seed=None, optimize=False, optimizeVars=None)

model = Sequential()
model.add(SimpleRNN(units=15, activation='relu', input_shape=(None, inputDim), return_sequences=True))
model.add(Dense(units=inputDim, activation='linear'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

startTime = time.time()
history = model.fit(logReturnsNormalized[:, np.newaxis, :], logReturnsNormalized, epochs=1000, batch_size=256, shuffle=False,
                    validation_split=0.2, callbacks=[earlyStopping])
endTime = time.time() 

encodedLogReturnsNormalized = model.predict(logReturnsNormalized[:, np.newaxis, :])
encodedLogReturnsNormalized = np.squeeze(encodedLogReturnsNormalized)
encodedLogReturns = scaler.inverse_transform(encodedLogReturnsNormalized)


reconstructionErrors = np.mean(np.square(logReturnsNormalized - encodedLogReturnsNormalized), axis=1)

meanError = np.mean(reconstructionErrors)
threshold = meanError + 3 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

elapsedTime = endTime - startTime
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
