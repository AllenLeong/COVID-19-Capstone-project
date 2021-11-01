import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime as dt
import import_ipynb
from Model_dev import pre_processing, data_handling, rmse

social_media = ['like_index','retweet_index']
covid_cases = ['ConfirmedCases', 'ConfirmedDeaths', 'Daily_cases']
general_info = ['CountryCode_x', 'CountryName_x', 'Jurisdiction', 'Date']
num_variable = ['E3_Fiscal measures', 'E4_International support', 'H5_Investment in vaccines', 'H4_Emergency investment in healthcare']
required_days = 14
pred_days = 7



# def data_handling(df, features):
    
#     #print("##### Reshape and MinMaxScale input #####")       
#     scaler = MinMaxScaler()

#     if 'Daily_cases' not in features:
#         features.append('Daily_cases')
#         df[features] = scaler.fit_transform(df[features])
#         features.remove('Daily_cases')       
#     else:
#         df[features] = scaler.fit_transform(df[features])
    
#     X = df[features]
#     y = df[['Daily_cases']]
#     X, y = pre_processing(X, y, required_days, pred_days, features)
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, shuffle=False, stratify = None)
#     #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#     return x_train, x_test, y_train, y_test, scaler


def draw_graph(y_pred_train, y_pred_test, y_true_train, y_true_test, input_days, output_days, country, features, x_train, algo, Date ,step,save=False):
    
    y_pred_train = y_pred_train.ravel()
    y_true_train = y_true_train.ravel()    
    y_true_test = y_true_test.ravel()
    y_pred_test = y_pred_test.ravel()

    plt.figure(figsize = (14,4))
    #Date = Date.astype(np.datetime64)

    Date = [dt.strptime(x, "%d/%m/%Y") for x in Date.values]
    plt.plot(Date[input_days:][120:len(y_true_train)],y_true_train[120:],label= 'Train Data',linewidth = 3,alpha=.4)
    plt.plot(Date[(len(y_true_train)+input_days):][:len(y_true_test)] ,y_true_test, '--',label ='Test Data',alpha=.4,linewidth = 3)
    
    plt.plot(Date[step+input_days+120:-output_days+1], np.delete(np.append(y_pred_train,y_pred_test), range(-step,-0))[120:], label = 'Prediction')
    plt.legend()
    plt.suptitle("{}th day predictions on {}'s daily confirmed cases using {} & {}".format(step+1,country, features, algo), fontsize=18)
    if save:
        plt.savefig('Picture/{}days_Prediction_{}_{}_{}.png'.format(step,algo, country, features))
    plt.show()
    
    
def load_dateset(country,f_flag, required_days, pred_days):
    filename = "Data/Full_{}.csv".format(country)
    df = pd.read_csv(filename, index_col=0)
    df.set_index('Date', inplace=True)
    policy = []
    columns = df.columns
    for column in columns:
        if not any(column in _list for _list in [social_media, general_info, covid_cases]):
            policy.append(column)
    categorical_variable = list(set(policy) - set(num_variable))
    df[categorical_variable] = df[categorical_variable].astype("category")
    
    feature_map = dict(P=policy, PC=policy+covid_cases, PCS=policy+covid_cases+social_media)
    features = feature_map[f_flag]
    Date = df.index
    x_train, x_test, y_train, y_test, scaler,features = data_handling(df, features, required_days, pred_days)
    
    return x_train, x_test, y_train, y_test, Date, scaler, features

def load_model_predict(model_path,x_train,x_test,y_train,y_test,step,scaler,features,custom_objects=None):
    loaded_model = models.load_model(model_path,custom_objects=custom_objects,compile=False)
    y_pred_train = loaded_model.predict(x_train)
    y_pred_test = loaded_model.predict(x_test)
    
    y_true_train = y_train[:,step,:]
    y_pred_train = y_pred_train[:,step,:]
    y_true_test = y_test[:,step,:]
    y_pred_test = y_pred_test[:,step,:]
    
    label_idx = np.where(np.array(features)=='Daily_cases')[0]
    if scaler is not None:
        y_true_train = y_inverse_scaler(y_true_train, scaler, label_idx)
        y_pred_train = y_inverse_scaler(y_pred_train, scaler, label_idx)
        y_true_test = y_inverse_scaler(y_true_test, scaler, label_idx)
        y_pred_test = y_inverse_scaler(y_pred_test, scaler, label_idx)
        
    return y_true_train, y_pred_train, y_true_test, y_pred_test

def y_inverse_scaler(y, scaler,label_idx):
    y_tile = np.tile(y,scaler.n_features_in_)
    return scaler.inverse_transform(y_tile)[:,label_idx]


def plot_predict(algo,model_path, f_flag,country, required_days = 14, pred_days = 7, step = 0,save=False,custom_objects=None):
    x_train, x_test, y_train, y_test, Date, scaler, features = load_dateset(country,f_flag,required_days, pred_days)
    y_true_train, y_pred_train, y_true_test, y_pred_test = load_model_predict(model_path,x_train,x_test,y_train,y_test,step,scaler,features,custom_objects)    
    draw_graph(y_pred_train, y_pred_test, y_true_train, y_true_test, required_days, pred_days, country, f_flag, x_train, algo, Date, step,save)
    
    
def plot_from_dataframe(df_log, index=0, step=0,save = False, custom_objects=None):
    algo = df_log.iloc[index,:]['Algorithm']
    model_path = df_log.iloc[index,:]['Model_path']
    f_flag = df_log.iloc[index,:]['Features']
    country = df_log.iloc[index,:]['Country']
    required_days = 14
    pred_days = 7
    plot_predict(algo,model_path, f_flag, country, required_days, pred_days,step,save, custom_objects) 
    return (algo, model_path, f_flag, country)
    
    
    
