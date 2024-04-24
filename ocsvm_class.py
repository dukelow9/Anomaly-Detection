

from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ROCAUC import plotROCAUC
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import roc_curve, auc

class Ocsvm:
    def __init__(self, currency, data):
        self.currency = currency
        self.data = data
        self.data_currency = self.data[self.data['Currency'] == self.currency].reset_index(drop=True)
        

    def train_test_split(self):

        # This function splits the data so that the data for 2022 is used as testing
        
        self.data_currency['Date'] = pd.to_datetime(self.data_currency['Date'], format="%d-%b-%y")
        df = self.data_currency.set_index(self.data_currency['Date'])
        df = df.sort_index()
        train = df[:'2021-12-31'].reset_index(drop=True)
        test  = df['2022-01-01':].reset_index(drop=True)
        
        X_train = train.iloc[:, 5:]
        X_test = test.iloc[:, 5:]
        Y_train = train['Anomaly Flag']
        Y_test = test['Anomaly Flag']
        return X_train, X_test, pd.DataFrame(Y_train), pd.DataFrame(Y_test)

    def scale_data(self, data):

        # This function applies the standard scaler to the data
        # returns the scaled data as a dataframe
        
        scaler_strd = StandardScaler()
        scaled_data = scaler_strd.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data)
        return scaled_df

    def differentiate(self, data):

        # This function differentiates the data
        #returns the differentaited data as a dataframe
        
        scaler = StandardScaler()
        diff_data = np.diff(data)
        scaled_diff = scaler.fit_transform(diff_data)
        diff_df = pd.DataFrame(scaled_diff)
        return diff_df
        
    def fit_ocsvm(self, data_df, nu, kernel, gamma):

        # fits the one class svm to the data and predicts on the training data
        # returns the dataframe with a column with the anomaly label, the index of the anomalous values and the model
        
        anom_df = pd.DataFrame(data_df).copy()
        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        model.fit(anom_df) #train data
        #Create a column for the anomaly
        anom_df['anomaly'] = pd.Series(model.predict(anom_df)).map({1: False, -1: True})
        anom_index = np.where(anom_df['anomaly'] == True)[0]
        return anom_df, anom_index, model

    def pred_ocsvm(self, data):

        # this function predicts the anomalies on the test data
        # returns the predictions and the indexes of the anomalous data pts
        
        predictions = model_diff.predict(pd.DataFrame(data))
        pred_index = np.where(predictions == -1)[0]
        return predictions, pred_index

    def sliding_window(self, data, window, nu, kernel, gamma):

        # applies a sliding window, the data is split into windows, and each window is fit to the ocsvm
        # returns the index of the anomalous points and the model
        
        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        anomalies = []
    
        for i in range(0, len(data) - window + 1, window):
            window_data = data.iloc[i:i + window, :]
            model.fit(window_data)
    
            predictions = pd.Series(model.predict(window_data)).map({1: False, -1: True})
    
            local_anom_index = np.where(predictions == True)[0]  # find index of anomalies within window
    
            global_anom_index = local_anom_index + i  # converts to global indexes to fit whole data
            anomalies.extend(global_anom_index.tolist())
    
        return anomalies, model

    def common_anom(self, detected_anom_index, labelled_anom):

        # finds the true anomalies which have been detected by the ocsvm
        
        true_anom = labelled_anom[labelled_anom['Anomaly Flag'] == 1]
        true_anom_index = true_anom.index.values
        common_anoms= np.intersect1d(true_anom_index, detected_anom_index)
        
        return common_anoms, true_anom_index

    def plot_ocsvm_roc(self, model, X, Y, title):

        # plots the roc of the ocsvm based on the decision functions
        
        scores = -model.decision_function(X)
        plotROCAUC(np.array(Y), scores)

    def best_auc(self, features, labels):

        # finds the parameters which give the best auc
        # returns best parameters and the best auc
        
        start = time.time()
        param_grid = {
            'gamma': [0.1, 0.01, 0.001, 0.05, 0.05, 0.005, 0.0001],
            'nu': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
        
        grid = ParameterGrid(param_grid)
        
        best_score = -1
        best_params = None
        best_recall = 0
        best_precision = 0
        
        # Iterate through parameter grid
        for params in grid:
            df, index, mod = self.fit_ocsvm(pd.DataFrame(features).copy(), params['nu'],  params['kernel'],  params['gamma'])
            # calc auc
            scores_tune = -mod.decision_function(pd.DataFrame(features))
            fpr, tpr, thresholds = roc_curve(np.array(labels), scores_tune)
            AUC = auc(fpr, tpr)
        
            # Update best parameters based on auc score
            if AUC > best_score:
                best_score = AUC
                best_params = params
        
        end = time.time()
        print('time elapsed: ', end - start)
        
        return best_params, best_score

    def best_f1(self, features, labels):

        # finds the parameters which give the best f1 score
        # returns bets parameters and best f1 score
        
        start = time.time()
        param_grid = {
            'gamma': [0.1, 0.01, 0.001, 0.05, 0.05, 0.005, 0.0001],
            'nu': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
        
        grid = ParameterGrid(param_grid)
        
        best_score = -1
        best_params = None
        
        # Iterate through parameter grid
        for params in grid:
            df, index, mod = self.fit_ocsvm(pd.DataFrame(features).copy(), params['nu'],  params['kernel'],  params['gamma'])
            common, true_index = self.common_anom(index, labels)
            # calc f1
            precision = len(common) / len(index)
            recall = len(common) / len(true_index)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
            # Update parameters based on F1 score
            if f1_score > best_score:
                best_score = f1_score
                best_params = params
                
        end = time.time()
        print('time elapsed: ', end - start)
        return best_params, best_score

data_path = ""
data = pd.read_csv(data_path)


currency = 'CAD'
svm = Ocsvm(currency, data)
X_train, X_test, Y_train, Y_test = svm.train_test_split()
print(X_train.shape, X_test.shape)
X_train_scaled = svm.scale_data(X_train)
X_test_scaled = svm.scale_data(X_test)
X_train_diff = svm.differentiate(X_train)
X_test_diff = svm.differentiate(X_test)

f1_params, f1 = svm.best_f1(X_train_scaled, Y_train)
print(f1, f1_params)

anom_df, anom_index, model = svm.fit_ocsvm(X_train_scaled, f1_params['nu'], f1_params['kernel'], f1_params['gamma'])
common, true = svm.common_anom(anom_index, Y_train)
print('true anoms that have been detected', len(common))
print('total anomalies detected', len(anom_index))
print('number of true anoms', len(true))
svm.plot_ocsvm_roc(model, X_train_scaled, Y_train)



