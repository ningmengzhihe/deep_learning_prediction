#!/usr/bin/env python
# coding: utf-8

# # 汇总conv1d/conv2d/LSTM/RNN/GRU/BiLSTM模型，多次迭代取预测结果的平均值
# Load Packages
# This makes it so that matplotlib graphics will show up within the Jupyter Notebook.
# Standard library import
import os
# Data Analysis Tools
import pandas as pd
import numpy as np
# Machine Learning Tools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
# File Loading Tools
import pickle
# Deep Learning Tools
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Dropout
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv2D, MaxPool2D
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


# In[2]:
# Load Data
X_train = np.load("../data/X_train_r_modeI_chamber4_mm.npy")
y_train = np.load("../data/y_train_modeI_chamber4_mm.npy")
X_test = np.load("../data/X_test_r_modeI_chamber4_mm.npy")
y_test = np.load("../data/y_test_modeI_chamber4_mm.npy")
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)

# Basic parameter
wafer_number, max_batch_length, variable_number = X_train.shape
wafer_number_test = X_test.shape[0]
print('训练集晶圆个数：', wafer_number)
print('最长时间序列长度：', max_batch_length)
print('字段个数：', variable_number)
print('训练集晶圆个数：', wafer_number_test)


# In[3]:
# reshape X for conv2d
X_train_r = X_train.reshape((wafer_number, max_batch_length, variable_number, 1))
X_test_r = X_test.reshape((wafer_number_test, max_batch_length, variable_number, 1))
print('X_train_r shape: ', X_train_r.shape)
print('X_test_r shape: ', X_test_r.shape)

# y value standarlization
ss = StandardScaler()
y_train_ss = ss.fit_transform(y_train)


# In[4]:
# conv1d/conv2d/LSTM/RNN/GRU/BiLSTM Model
def predict_conv1d():
    # Step1:define net
    model = Sequential(name='model_conv1d')
    model.add(Convolution1D(filters=256, kernel_size=2, activation='relu', input_shape=(max_batch_length, variable_number), name='conv1d1'))
    model.add(MaxPooling1D(8, name='maxpool1'))
    model.add(Convolution1D(filters=128, kernel_size=2, activation='relu', name='conv1d2'))
    model.add(MaxPooling1D(8, name='maxpool2'))
    model.add(Convolution1D(filters=64, kernel_size=2, activation='relu', name='conv1d3'))
    model.add(MaxPooling1D(2, name='maxpool3'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(16, activation='linear', name='dense1'))
    model.add(Dense(1, activation='linear', name='dense2'))
    print('Conv1d model define done')
        
    # Step2:compile
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # Step3:fit
    history_model_conv1d = model.fit(X_train, y_train, epochs=400, batch_size=200) #epoch400
    print('Conv1d model fit done')
    
    # Step4:predict test set
    y_test_pre = model.predict(X_test)
    print('Predict test set done')
    
    # Step5:compute metrics
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
    mae = mean_absolute_error(y_test, y_test_pre)
    r2 = r2_score(y_test, y_test_pre)
    print('Compute metrics done')
    
    return y_test_pre, mse, rmse, mae, r2, model


def predict_conv2d():
    # Step1:define model
    model = Sequential(name='model_conv2d')

    model.add(Conv2D(filters=16, kernel_size=[2, 2], padding='same', input_shape=(max_batch_length, variable_number, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=32, kernel_size=[2, 2], padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=64, kernel_size=[2, 2], padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=16, kernel_size=[2, 2], padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))
    
    print('Conv2d model define done')
    
    # Step2:compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    nb_epoch = 1000
    batch_size = 399

    Reduce=ReduceLROnPlateau(monitor='mse',
                             factor=0.5,
                             patience=10,
                             verbose=1,
                             mode='min',
                             min_delta=0.0001,
                             cooldown=0,
                             min_lr=1e-10)
    
    # Step3:fit
    model.fit(x=X_train_r, y=y_train, epochs=nb_epoch, batch_size=batch_size, callbacks=[Reduce])
    print('Conv1d model fit done')
    
    # Step4:predict test set
    y_test_pre = model.predict(X_test_r)
    print('Predict test set done')
    
    # Step5:compute metrics
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
    mae = mean_absolute_error(y_test, y_test_pre)
    r2 = r2_score(y_test, y_test_pre)
    print('Compute metrics done')
    
    return y_test_pre, mse, rmse, mae, r2, model


def predict_lstm():
    # Step1:define model
    model_lstm = Sequential(name='model_lstm')
    model_lstm.add(LSTM(input_shape=(max_batch_length,variable_number), units=40, activation='tanh', return_sequences=True, name='lstm1'))
    model_lstm.add(LSTM(units=40, activation='tanh', return_sequences=False, name='lstm2'))

    model_lstm.add(Dense(4, name='dense1'))
    model_lstm.add(Dense(1, name='dense2'))
    
    print('LSTM model define done')
    
    # Step2:compile model
    adam = Adam(learning_rate=0.001)
    model_lstm.compile(loss='mse', optimizer=adam, metrics=['mse'])

    nb_epoch = 400
    batch_size = 200
    
    # Step3:fit
    model_lstm.fit(X_train, y_train_ss, epochs=nb_epoch, batch_size=batch_size)
    print('LSTM model fit done')
    
    # Step4:predict test set
    y_test_pre = model_lstm.predict(X_test)
    y_test_pre = ss.inverse_transform(y_test_pre)
    print('Predict test set done')
    
    # Step5:compute metrics
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
    mae = mean_absolute_error(y_test, y_test_pre)
    r2 = r2_score(y_test, y_test_pre)
    print('Compute metrics done')
    
    return y_test_pre, mse, rmse, mae, r2, model_lstm


def predict_rnn():
    # Step1:define model
    model_rnn = Sequential(name='model_rnn')
    model_rnn.add(SimpleRNN(input_shape=(max_batch_length,variable_number), units=40, activation='tanh', return_sequences=True))
    model_rnn.add(SimpleRNN(units=40, activation='tanh', return_sequences=False))

    model_rnn.add(Dense(4))
    model_rnn.add(Dense(1))
    print('RNN model define done')
    
    # Step2:compile model
    adam = Adam(learning_rate=0.001)
    model_rnn.compile(loss='mse', optimizer=adam, metrics=['mse']) # rmsprop

    nb_epoch = 100
    batch_size = 200
    
    # Step3:fit
    model_rnn.fit(X_train, y_train_ss, epochs=nb_epoch, batch_size=batch_size)
    print('RNN model fit done')
    
    # Step4:predict test set
    y_test_pre = model_rnn.predict(X_test)
    y_test_pre = ss.inverse_transform(y_test_pre)
    print('Predict test set done')
    
    # Step5:compute metrics
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
    mae = mean_absolute_error(y_test, y_test_pre)
    r2 = r2_score(y_test, y_test_pre)
    print('Compute metrics done')
    
    return y_test_pre, mse, rmse, mae, r2, model_rnn


def predict_gru():
    # Step1:define model
    model_gru = Sequential(name='model_GRU')
    model_gru.add(GRU(input_shape=(max_batch_length,variable_number), units=40, activation='relu', return_sequences=True))
    model_gru.add(GRU(units=40, activation='relu', return_sequences=False))

    model_gru.add(Dense(4))
    model_gru.add(Dense(1))
    
    print('GRU model define done')
    
    # Step2:compile model
    adam = Adam(learning_rate=0.001)
    model_gru.compile(loss='mse', optimizer=adam, metrics=['mse']) # rmsprop

    nb_epoch = 50 #150
    batch_size = 100

    # Step3:fit
    model_gru.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size)
    print('GRU model fit done')
    
    # Step4:predict test set
    y_test_pre = model_gru.predict(X_test)
    print('Predict test set done')
    
    # Step5:compute metrics
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
    mae = mean_absolute_error(y_test, y_test_pre)
    r2 = r2_score(y_test, y_test_pre)
    print('Compute metrics done')
    
    return y_test_pre, mse, rmse, mae, r2, model_gru


def predict_bilstm():
    # Step1:define model
    model_biLSTM = Sequential(name='model_biLSTM')
    model_biLSTM.add(Bidirectional(LSTM(units=40, activation='tanh', return_sequences=True), input_shape=(max_batch_length,variable_number), name='biLSTM1'))
    model_biLSTM.add(Bidirectional(LSTM(units=40, activation='tanh', return_sequences=False), name='biLSTM2'))

    model_biLSTM.add(Dense(4, name='dense1'))
    model_biLSTM.add(Dense(1, name='dense2'))

    print('BiLSTM model define done')
    
    # Step2:compile model
    adam = Adam(learning_rate=0.001)
    model_biLSTM.compile(loss='mse', optimizer=adam, metrics=['mse'])
    nb_epoch = 50
    batch_size = 200

    # Step3:fit
    model_biLSTM.fit(X_train, y_train_ss, epochs=nb_epoch, batch_size=batch_size)
    print('BiLSTM model fit done')
    
    # Step4:predict test set
    y_test_pre = model_biLSTM.predict(X_test)
    y_test_pre = ss.inverse_transform(y_test_pre)
    print('Predict test set done')
    
    # Step5:compute metrics
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
    mae = mean_absolute_error(y_test, y_test_pre)
    r2 = r2_score(y_test, y_test_pre)
    print('Compute metrics done')
    
    return y_test_pre, mse, rmse, mae, r2, model_biLSTM


def deep_learning_iteration(model_name, N_iter=10):
    y_test_pre_list = []
    mse_list, rmse_list, mae_list, r2_list = [], [], [], []
    print('*******************')
    for i in range(0, N_iter):
        print(f'Iter {i} Start:')
        y_test_pre, mse, rmse, mae, r2, model = eval(f'predict_{model_name}')()

        y_test_pre_list.append(y_test_pre)
        mse_list.append(mse); rmse_list.append(rmse); mae_list.append(mae); r2_list.append(r2)

    y_test_pre_mean = np.mean(y_test_pre_list, axis=0)
    mse_mean, rmse_mean, mae_mean, r2_mean = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(r2_list)

    print(f'Mean results of {N_iter} times iteration:')
    print(f'Mean mse of {N_iter} times iteration:', mse_mean)
    print(f'Mean rmse of {N_iter} times iteration:', rmse_mean)
    print(f'Mean mae of {N_iter} times iteration:', mae)
    print(f'Mean r2 of {N_iter} times iteration:', r2_mean)

    dict_result = {}
    dict_result = {'y_test_pre': y_test_pre_mean, 'mse': mse_mean, 'rmse': rmse_mean, 'mae': mae_mean, 'r2': r2_mean}

    pickle.dump(dict_result, open(f'../results_save/results_{model_name}.pkl', 'wb'))
    # save last model
    model.save(f'../results_save/model_{model_name}.h5')


# In[5]:
# Deep Learning Model iteration
# deep_learning_iteration(model_name='conv1d', N_iter=10)
# deep_learning_iteration(model_name='conv2d', N_iter=10)
# deep_learning_iteration(model_name='lstm', N_iter=10)
# deep_learning_iteration(model_name='rnn', N_iter=10)
deep_learning_iteration(model_name='gru', N_iter=10)
# deep_learning_iteration(model_name='bilstm', N_iter=10)





