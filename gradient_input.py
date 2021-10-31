import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, activations, models, metrics, initializers, Model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime as dt
import import_ipynb
from Model_dev import compress_to_2d, draw_graph, pre_processing, plot_history, data_handling, rmse
import prediction_painter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse


class GradientInput():
    """
    class GradientInput
    """
    def __init__(self, algo, model_path, f_flag, country, custom_objects = None):
        """
        Call:
            Arg:
                :algo: algorithm (C-LSTM, Attention, LSTM),
                :model_path: path to model,
                :f_flag: "P", "PC", "PSC",
                :country:
        """
        self.algo = algo
        self.model_path = model_path
        self.f_flag = f_flag
        self.country = country
        self.custom_objects = custom_objects
        # load dateset
        x_train, x_test, y_train, y_test, Date, self.scaler, self.features = prediction_painter.load_dateset(country, f_flag, 14,7)
        # load model
        self.loaded_model = models.load_model(model_path, custom_objects = custom_objects,compile=False)

        # merge train and test into one
        self.X = np.concatenate((x_train, x_test),axis=0)
        self.y = np.concatenate((y_train, y_test),axis=0)
        self.DateX = Date[:self.X.shape[0]+14]
        self.Datey = Date[14:self.X.shape[0]+14+7]
        print(self.algo, model_path, self.f_flag, self.country)
        print("X shape:",self.X.shape, " y shape:",self.y.shape)
        print('Date Range:{}~{}'.format(self.DateX[0], self.DateX[-1]))

    def gradient_input(self, index, ratio = 0.5, step = 1000, learning_rate = 5e-4,gradient_only=False):
        """
        Func gradient input to index slice
            Arg:
                :index: str "31/4/2020"  or int 303
                :ratio: decrease by ratio
                :step: approximating step
                :learning_rate:
            return: df_gradient DateFrame. gradiented indexed slice

        """
        if type(index) == str: index = np.where(np.array(self.DateX) == index)[0][0]

        A_orig = A = self.X[index]
        A_Date = self.DateX[index:index+14]
        y_Date = self.Datey[index:index+7]
        A_orig = tf.expand_dims(tf.convert_to_tensor(A_orig),0)

        Y_target = self.loaded_model.predict(A_orig)* ratio # decrease the target by ratio
        mse = tf.keras.losses.MeanSquaredError()
        A = tf.expand_dims(A,0)
        # mask those features that are out of arbitration
        mask_id = np.where([i in
                  ['ConfirmedCases', 'ConfirmedDeaths', 'Daily_cases', 'like_index','retweet_index']
                  for i in np.array(self.features)])
        losses = []
        A = self.gradient_tape(A,Y_target,mask_id,step, learning_rate,gradient_only) # get new A
        if gradient_only:
            return pd.DataFrame(A[0].numpy().T,
                           columns = A_Date,
                           index = self.features)

        self.plot_new_predictions(A_orig,A ,A_Date,y_Date, Y_target)
        A_divid = A_orig.numpy()
        A_divid[A_divid==0] = 1

        df_gradient = pd.DataFrame((A[0].numpy().T-A_orig.numpy()[0].T)/A_divid[0].T,
                           columns = A_Date,
                           index = self.features)
        self.df_gradient = df_gradient
        return df_gradient

    def gradient_tape(self, A,Y_target,mask_id=False,step = 1000, learning_rate = 5e-4, gradient_only = False):
        """
        Func tf.GradientTape Process
            Arg:
                :A: tf.Tensor: Input
                :step: approximation step
                :learning_rate:
                :gradient_only: bool, return gradient only 
            return: tf.Tensor, updated A
        """
        loss_record = []
        mse = tf.keras.losses.MeanSquaredError()
        for i in tqdm(range(step)):
            with tf.GradientTape() as tape:
                tape.watch(A)
                Y_hat = self.loaded_model(A)

                loss = Y_target - Y_hat # (1,7,1)
                #loss.numpy().mean()
                loss_record.append(mse(Y_target, Y_hat).numpy())
                if i > 0 and loss_record[-1] >= loss_record[-2]:
                    loss_record = loss_record[:-1]
                    break
                gradient = tape.gradient(loss,A)
                gradient = gradient.numpy()
                
                if gradient_only:
                    return tf.convert_to_tensor(gradient)
                
                if mask_id:
                    for mask in mask_id[0]:
                        gradient[:,:,mask] = np.zeros(gradient.shape[1])
                gradient = tf.convert_to_tensor(gradient)
                #plt.imshow(gradient[0,:,:])
                #sns.heatmap(gradient[0,:,:], linewidth=0.5)
                A = A+gradient*learning_rate
                A = A.numpy()
                A[A < 0] = 0
                A = tf.convert_to_tensor(A)
        # plot loss
        self.loss_record = loss_record
        plt.figure(figsize=(12,4))
        plt.plot(loss_record);plt.title("Loss")
        plt.show()
        return A

    def plot_guided_input(self):
        # plot A after gradients
        plt.figure(figsize=(12,6))
        sns.heatmap(self.df_gradient, cmap = plt.cm.PiYG, center = 0)
        #plt.yticks(range(x_train.shape[2]), features)
        plt.tight_layout()
        plt.show()

    def plot_new_predictions(self,A_orig,A, A_Date, y_Date, Y_target):
        y_orig = self.loaded_model.predict(A_orig)
        y_after = self.loaded_model.predict(A)

        daily_cases_id = np.where(np.array(self.features) == 'Daily_cases')[0]
        daily_cases_X = A.numpy()[0,:,daily_cases_id]

        plt.figure(figsize = (12,4))
        plt.scatter([dt.strptime(x, "%d/%m/%Y") for x in A_Date.values],
                     prediction_painter.y_inverse_scaler(daily_cases_X.T,self.scaler, daily_cases_id),
                     s=150,edgecolors='black',color='white')
        plt.scatter([dt.strptime(x, "%d/%m/%Y") for x in A_Date.values],
                    prediction_painter.y_inverse_scaler(daily_cases_X.T,self.scaler, daily_cases_id),
                    label = 'Inputs',s=60,edgecolors='black')
        plt.scatter([dt.strptime(x, "%d/%m/%Y") for x in y_Date.values],
                    prediction_painter.y_inverse_scaler(y_orig[0],self.scaler, daily_cases_id),
                    label = 'Original predictions',s=80,edgecolors='black',color = 'lightgrey')
        plt.scatter([dt.strptime(x, "%d/%m/%Y") for x in y_Date.values],
                    prediction_painter.y_inverse_scaler(y_after[0],self.scaler, daily_cases_id),
                    label = 'Post-gradient predictions',s=80,edgecolors='black',color = 'grey')
        plt.scatter([dt.strptime(x, "%d/%m/%Y") for x in y_Date.values],
                    prediction_painter.y_inverse_scaler(Y_target[0],self.scaler, daily_cases_id),
                    label = 'Target predictions',s=80,edgecolors='black',color = 'green',marker = 'v',alpha=.6)
        plt.legend()
        plt.title('Original Output vs Target Output')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", help="Do gradient", action='store_true')
    parser.add_argument("-p", help="Plot gradient",action='store_true')
    parser.add_argument("-u", help="User Object",action='store_true')
    parser.add_argument("--i", help="log file index",type=int)
    parser.add_argument("--r", help="log file index",type=float)
    parser.add_argument("--d", help="Date",type=str)
    args = parser.parse_args()

    df_log = pd.read_csv("Log/Models.csv", index_col=0)
    df_log['Model_path'] = df_log['Model_path'].apply(lambda x: x.replace('C:\\Users\\wasin\\Downloads\\Work\\PG(HKU)\\FYP\\Program_Data\\',''))

    custom_objects = dict(rmse= rmse) if args.u else None

    algo, model_path, f_flag, country = prediction_painter.plot_from_dataframe(df_log,args.i,custom_objects= custom_objects)

    if args.g:
        gradient_trainer = GradientInput(algo, model_path, f_flag, country)
        gradient_trainer.gradient_input(args.d, ratio = args.r, step = 1000, learning_rate = 5e-4)
    if args.p:
        gradient_trainer.plot_guided_input()
