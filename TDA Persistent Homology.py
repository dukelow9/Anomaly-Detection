#all the required packages
from sklearn.metrics import roc_curve, auc
import latex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cython
from ripser import Rips
import persim
import time


# dunno if this is required
plt.rcParams.update(plt.rcParamsDefault)


df = pd.read_csv('1331 1453 Dataset 2 With Anomalies.txt')
rips = Rips(maxdim = 2) # Defines the maximum number of dimensions to use for persistent homology


# Everything redone in functions
def datasetPreparation(currencyOfInterest):

    dataOI = df[df['Currency'] == currencyOfInterest]
    dates = pd.to_datetime(dataOI['Date'])
    spotRate = dataOI['Spot Rate']

    allRates = dataOI.iloc[:, 5:21]  # Goes from 5 till 21   5:9 means spot,1w,2w,3w,1M
    allRates = allRates.where(allRates > 0.0)  # replaces all -1 values with NAN to be removed they can be removed
    allRates = allRates.dropna(axis=1, thresh=100)  # Drops columns where there are at least 100 NAN values
    allRates = allRates.dropna(axis=0)  # Drops all rows with NAN values

    print(allRates)
    allRatesWTime = allRates
    allRatesWTime['Date'] = dates
    allRatesWTime = allRatesWTime.set_index('Date')
    logRatesWT = np.log(allRatesWTime / allRatesWTime.shift(1))



    return logRatesWT, dates, dataOI, spotRate


def wassersteinDistCalculator(windowSize,logRatesWT):

    numberOfWindows = len(logRatesWT) - (2 * windowSize) + 1
    wassersteinDistances = np.zeros(numberOfWindows)
    start = time.time()
    for i in range(1, numberOfWindows):
        distribution1 = rips.fit_transform(logRatesWT[i:i + windowSize])
        distribution2 = rips.fit_transform(logRatesWT[i + windowSize + 1:i + (2 * windowSize) + 1])

        wassersteinDistances[i] = persim.wasserstein(distribution1[0], distribution2[0], matching=False)
    end = time.time()

    # Calculating the Derivatives of Wasserstein Distances
    wassersteinDerivatives = np.zeros(len(wassersteinDistances) - 1)
    for i in range(len(wassersteinDistances) - 1):
        wassersteinDerivatives[i] = (wassersteinDistances[i + 1] - wassersteinDistances[i]) / 1

    return wassersteinDistances, wassersteinDerivatives, numberOfWindows


def plottingPrepforDofWSD(dataOI, spotRate, dates, windowSize, numberOfWindows, wassersteinDerivatives):
    x = dates[windowSize:numberOfWindows + windowSize - 1]  # -1 here for D of WSD
    sptRteForGraph = spotRate[windowSize:numberOfWindows + windowSize - 1]  # -1 here for D of WSD
    flaggedAnoms = dataOI['Anomaly Flag']
    flaggedAnoms = flaggedAnoms[windowSize:numberOfWindows + windowSize - 1]  # adding -1 so it works with D of WSD
    # Setting up a data frame with correctly sized columns for the windows

    df2 = pd.DataFrame({'Date': x})
    df2['D of Wasserstein Distances'] = wassersteinDerivatives
    df2['Spot Rate'] = sptRteForGraph
    df2['Anomaly Flag'] = flaggedAnoms
    df2 = df2.set_index('Date')

    return df2, sptRteForGraph, x

def plottingPrepforWSD(dataOI, spotRate, dates, windowSize, numberOfWindows, wassersteinDerivatives):
    x = dates[windowSize:numberOfWindows + windowSize]
    sptRteForGraph = spotRate[windowSize:numberOfWindows + windowSize]
    flaggedAnoms = dataOI['Anomaly Flag']
    flaggedAnoms = flaggedAnoms[windowSize:numberOfWindows + windowSize]
    # Setting up a data frame with correctly sized columns for the windows

    df2 = pd.DataFrame({'Date': x})
    #df2['D of Wasserstein Distances'] = wassersteinDerivatives
    df2['Spot Rate'] = sptRteForGraph
    df2['Anomaly Flag'] = flaggedAnoms
    df2 = df2.set_index('Date')

    return df2, sptRteForGraph, x


def plotROCAUC(labels, reconstructionErrors, currencyOfInterest):
    fpr, tpr, thresholds = roc_curve(labels, reconstructionErrors)
    ROCAUC = auc(fpr, tpr)
    print(thresholds)

    plt.figure()
    plt.plot(fpr, tpr, color='lime', lw=2, label=f'AUC = {ROCAUC:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for WSD on {currencyOfInterest}')
    plt.legend(loc='lower right', )
    plt.grid()
    plt.show()


def WSDvsSpotRate(currencyOfInterest, x,wassersteinDerivatives ,sptRteForGraph):
    fig2, (ax1, ax2) = plt.subplots(2)
    fig2.suptitle(f'Persistent Homology with Derivative of Wasserstein Dists. for {currencyOfInterest} up to 10Y')
    ax1.set_ylabel(f'Spot Rate for {currencyOfInterest}')
    ax2.set_ylabel('Derivative of Wasserstein Distance')
    ax1.plot(x, sptRteForGraph, label='Spot Rate', color='r')
    ax2.plot(x, wassersteinDerivatives, label='Derivative of Wasserstein Distances', color='b')

    fig2.legend()
    plt.show()






def DofwassersteinDistPlot(currencyOfInterest,windowSize):
    lograteswt, dates, dataoi, spotrate = datasetPreparation(currencyOfInterest)

    start = time.time()
    wDistances, wDerivatives, numberOfWindows =  wassersteinDistCalculator(windowSize,lograteswt)
    end = time.time()
    print('Running the Wasserstein Distance took', end - start, 'seconds')

    df2, sptRteForGraph, x = plottingPrepforDofWSD(dataoi, spotrate, dates, windowSize, numberOfWindows, wDerivatives)

    plotROCAUC(df2['Anomaly Flag'], abs(wDerivatives), currencyOfInterest)

    WSDvsSpotRate(currencyOfInterest, x, abs(wDerivatives), sptRteForGraph)


def wassersteinDistPlot(currencyOfInterest,windowSize):
    lograteswt, dates, dataoi, spotrate = datasetPreparation(currencyOfInterest)

    start = time.time()
    wDistances, wDerivatives, numberOfWindows = wassersteinDistCalculator(windowSize, lograteswt)
    end = time.time()
    print('Running the Wasserstein Distance took', end - start, 'seconds')

    df2, sptRteForGraph, x = plottingPrepforWSD(dataoi, spotrate, dates, windowSize, numberOfWindows, wDerivatives)

    plotROCAUC(df2['Anomaly Flag'], wDistances, currencyOfInterest)

    WSDvsSpotRate(currencyOfInterest, x, wDistances, sptRteForGraph)


test1 = DofwassersteinDistPlot('GBP',50)

#test2 = wassersteinDistPlot('INR', 50)













'''
# Reading data, making a dataframe, plotting log returns

df = pd.read_csv('1331 1453 Dataset 2 With Anomalies.txt')
currencyOfInterest = 'GBP'
dataOI = df[df['Currency'] == currencyOfInterest]
dates = pd.to_datetime(dataOI['Date'])
spotRate = dataOI['Spot Rate']

# Multivariate Data
allRates = dataOI.iloc[:,5:21]  # Goes from 5 till 21   5:9 means spot,1w,2w,3w,1M
print(allRates)
allRatesWTime = allRates
allRatesWTime['Date'] = dates
allRatesWTime = allRatesWTime.set_index('Date')
logRatesWT = np.log(allRatesWTime/allRatesWTime.shift(1))
'''

# Rips Persistence diagrams

'''
dgm = rips.fit_transform(logRatesWT[1:750])
plt.figure(figsize=(5, 5), dpi=80)
persim.plot_diagrams(dgm, title=f"Persistence Diagram for {currencyOfInterest}")
plt.show()
'''




# Wasserstein Distance Calculation
'''
windowSize = 50
numberOfWindows = len(logRatesWT) - (2 * windowSize) + 1
wassersteinDistances = np.zeros(numberOfWindows)
start = time.time()
for i in range(1,numberOfWindows):
    distribution1 = rips.fit_transform(logRatesWT[i:i+windowSize])
    distribution2 = rips.fit_transform(logRatesWT[i+windowSize+1:i+(2*windowSize)+1])

    wassersteinDistances[i] = persim.wasserstein(distribution1[0],distribution2[0],matching=False)
end = time.time()
#print('Running the Wasserstein Distance took', end - start, 'seconds')


# Calculating the Derivatives of Wasserstein Distances
wassersteinDerivatives = np.zeros(len(wassersteinDistances)-1)
for i in range(len(wassersteinDistances)-1):
    wassersteinDerivatives[i] = (wassersteinDistances[i+1]-wassersteinDistances[i])/1
'''







# Wasserstein Distance Plot data prep

'''
# Setting up the correctly sized arrays with easy names for easier plotting
x = dates[windowSize:numberOfWindows+windowSize-1] #-1 here for D of WSD
sptRteForGraph = spotRate[windowSize:numberOfWindows+windowSize-1] #-1 here for D of WSD
flaggedAnoms = dataOI['Anomaly Flag']
flaggedAnoms = flaggedAnoms[windowSize:numberOfWindows+windowSize-1] # adding -1 so it works with D of WSD
# Setting up a data frame with correctly sized columns for the windows


df2 = pd.DataFrame({'Date': x})
df2['D of Wasserstein Distances'] = wassersteinDerivatives
df2['Spot Rate'] = sptRteForGraph
df2['Anomaly Flag'] = flaggedAnoms
df2 = df2.set_index('Date')

print(df2)
'''









#plotROCAUC(df2['Anomaly Flag'], abs(wassersteinDerivatives))
#plotROCAUC(flaggedAnoms, wassersteinDistances)

# Plotting both graphs on the same plot with two scales
'''
fig,WSD = plt.subplots()
SPTRT = WSD.twinx()
WSD.set_ylabel('Wasserstein Distance')
SPTRT.set_ylabel(f'Spot Rate for {currencyOfInterest}')
plot1 = WSD.plot(x, wassersteinDistances, label='Wasserstein Distances', color='b')
plot2 = SPTRT.plot(x, sptRteForGraph, label='Spot Rate', color='r')

fig.legend()
plt.title(f'Persistent Homology with Wasserstein Dists. for {currencyOfInterest}')
plt.plot()
plt.show()
'''
# Plotting both graphs one on top of the other
'''
fig2, (ax1, ax2) = plt.subplots(2)
fig2.suptitle(f'Persistent Homology with Wasserstein Dists. for {currencyOfInterest} up to 10Y')
ax1.set_ylabel(f'Spot Rate for {currencyOfInterest}')
ax2.set_ylabel('Wasserstein Distance')
ax1.plot(x, sptRteForGraph, label='Spot Rate', color='r')
ax2.plot(x, wassersteinDistances, label='Wasserstein Distances', color='b')

fig2.legend()
plt.show()
'''

# Plotting Derivative of Wasserstein Distances
'''
fig2, (ax1, ax2) = plt.subplots(2)
fig2.suptitle(f'Persistent Homology with Derivative of Wasserstein Dists. for {currencyOfInterest} up to 10Y')
ax1.set_ylabel(f'Spot Rate for {currencyOfInterest}')
ax2.set_ylabel('Derivative of Wasserstein Distance')
ax1.plot(x, sptRteForGraph, label='Spot Rate', color='r')
ax2.plot(x, wassersteinDerivatives, label='Derivative of Wasserstein Distances', color='b')

fig2.legend()
plt.show()
'''