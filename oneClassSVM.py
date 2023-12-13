from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ## Load data 


data_url = "C:/Users/Mahee/Documents/Group proj/Data/1128_1419_fx_tenor_curves_data.csv"
data = pd.read_csv(data_url)


# #### Make column for log returns of spot rate

data['Log Returns'] = np.log(data['Spot Rate'].shift(1)/data['Spot Rate'])


currency = 'GBP'
filteredData = data[data['Currency'] == currency] 
dates = filteredData['Date']
spot_rates = filteredData['Spot Rate']
spot_log_ret = filteredData['Log Returns']

# resets index no. so starts from 0 (when picking a currency index does not start from 0)
dates=dates.reset_index(drop=True) 
spot_rates=spot_rates.reset_index(drop=True)
spot_log_ret=spot_log_ret.reset_index(drop=True)

# ## One class SVM algorithm

def ocsvm( spot_rate_df, nu, kernel, gamma):
    spot_rates_flat = spot_rate_df.values.reshape(-1,1) # flatten data
    scaler = StandardScaler()
    spot_rates_scaled = scaler.fit_transform(spot_rates_flat) # scale data
    scaled_df = pd.DataFrame(spot_rates_scaled)

    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(scaled_df) # train data
    
    # Create a column for the anomaly
    spot_rate_df['anomaly'] = pd.Series(model.predict(scaled_df)).map({1: False, -1: True})
    return spot_rate_df


def plotAnomaly(df_anomaly, dates, spot_rates):
    plt.plot(dates, spot_rates)
    plt.xlabel("Dates")
    plt.ylabel("Spot Rate")
    plt.title( "Spot rate for GBP")
    anomalyDates=dates.loc[df_anomaly['anomaly']] # locates all dates where anomaly is true
    anomalySpots = spot_rates.loc[df_anomaly['anomaly']] # locates all spot rates where anomaly is true
    plt.scatter(anomalyDates, anomalySpots, color='red')
    plt.show()


# ## Results


df_anomaly_2 = ocsvm(spot_rates.copy(), 0.01, 'rbf', 0.01)
plotAnomaly(df_anomaly_2, dates, spot_rates)


# #### Log return results


df_anomaly_3 = ocsvm(spot_log_ret.copy(), 0.001, 'rbf', 0.01)
plotAnomaly(df_anomaly_3, dates, spot_rates)
plt.plot(dates[1:], spot_log_ret[1:])
plt.scatter(dates[1:].loc[df_anomaly_3['anomaly'][1:]], spot_log_ret[1:].loc[df_anomaly_3['anomaly'][1:]], color='red')
plt.show()



