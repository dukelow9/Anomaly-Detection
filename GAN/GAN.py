import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def createGenerator(latentDim, outputDim):
    model = Sequential()
    model.add(Dense(8, input_dim=latentDim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(outputDim, activation='linear'))
    return model

def createDiscriminator(inputDim):
    model = Sequential()
    model.add(Dense(16, input_dim=inputDim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def createGAN(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

data = pd.read_excel("C:\\Users\\User\\OneDrive\\Documents\\Notes\\Year 3\\GP\\data1.xlsx")
currency = 'GBP'
filteredData = data[data['Currency'] == currency]
dates = filteredData['Date']
rates = filteredData.loc[:, 'Spot Rate':'10Y Rate']

logReturns = np.log(rates / rates.shift(1)).dropna()

scaler = MinMaxScaler()
logReturnsNormalized = scaler.fit_transform(logReturns)

latentDim = 10

generator = createGenerator(latentDim, logReturnsNormalized.shape[1])
discriminator = createDiscriminator(logReturnsNormalized.shape[1])

discriminator.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')

gan = createGAN(generator, discriminator)
gan.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

startTime = time.time()

realDLossHistory = []
fakeDLossHistory = []
GLossHistory = []

for epoch in range(1000):
    noise = np.random.normal(0, 1, (len(logReturnsNormalized), latentDim))
    gData = generator.predict(noise)
    realData = logReturnsNormalized
    realLabel = np.ones((len(logReturnsNormalized), 1))
    fakeLabel = np.zeros((len(logReturnsNormalized), 1))

    realDLossValue = discriminator.train_on_batch(realData, realLabel)
    fakeDLossValue = discriminator.train_on_batch(gData, fakeLabel)
    DLoss = 0.5 * np.add(realDLossValue, fakeDLossValue)

    noise = np.random.normal(0, 1, (len(logReturnsNormalized), latentDim))
    validLabel = np.ones((len(logReturnsNormalized), 1))
    GLoss = gan.train_on_batch(noise, validLabel)

    realDLossHistory.append(realDLossValue)
    fakeDLossHistory.append(fakeDLossValue)
    GLossHistory.append(GLoss)

    print(f"Epoch {epoch}, D Loss: {DLoss}, G Loss: {GLoss}")

endTime = time.time()

noise = np.random.normal(0, 1, (len(logReturnsNormalized), latentDim))
gData = generator.predict(noise)
gData = scaler.inverse_transform(gData)

reconstructionErrors = np.mean(np.square(logReturnsNormalized - gData), axis=1)
meanError = np.mean(reconstructionErrors)
threshold = meanError + 3 * np.std(reconstructionErrors)
anomalies = np.where(reconstructionErrors > threshold)[0]

elapsedTime = endTime - startTime
print(f"Training Time: {elapsedTime} seconds")
