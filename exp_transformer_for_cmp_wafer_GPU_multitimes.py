#!/usr/bin/env python
# coding: utf-8

# # 处理CMP化学研磨过程数据[ok]

# Data Analysis Tools
import pandas as pd
import numpy as np

# Visualization Tools
import matplotlib.pyplot as plt
import seaborn as sns

import os
from tqdm import tqdm
import datetime
import pickle

# Deep Learning Tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

# Transformer for time series Class
from utils.transformers_tst.tst import Transformer
from utils.transformers_tst.src.utils import compute_loss
from utils.transformers_tst.src.visualization import map_plot_function, plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample

# Machine Learning Tools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ## 2. 封装成class类

# ## 2.1 导入模态I腔室4的训练和测试数据
X_train = np.load("../data/X_train_r_modeI_chamber4_mm.npy")
y_train = np.load("../data/y_train_modeI_chamber4_mm.npy")
X_test = np.load("../data/X_test_r_modeI_chamber4_mm.npy")
y_test = np.load("../data/y_test_modeI_chamber4_mm.npy")
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)

# 基本参数
wafer_number, max_batch_length, variable_number = X_train.shape
wafer_number_test = X_test.shape[0]
print('Wafer number of Training Set：', wafer_number)
print('Maximum BatchSize：', max_batch_length)
print('Number of variables：', variable_number)
print('Wafer number of Testing set：', wafer_number_test)


class CMPModeIChamber4Dataset(Dataset):
    """Torch dataset for Oze data challenge training.
    Attributes
    ----------
    x: np.array
        Dataset target of shape (wafer_number, seq_length, variable_number).
    
    y: np.array
        Dataset target of shape (wafer_number, 1).
    """

    def __init__(self, dataset_x, dataset_y, **kwargs):
        """Load dataset from csv.
        Parameters
        ---------
        dataset_x: Tuple
            Tuple of shape (wafer_number, seq_length, variable_number).
        dataset_y: Tuple
            Tuple of shape (wafer_number, 1).
        """
        super().__init__(**kwargs)

        self._x = dataset_x.astype(np.float32)
        self._y = dataset_y.astype(np.float32)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self._x[idx], self._y[idx])
    
    def __len__(self):
        return self._x.shape[0]

    def get_x_shape(self):
        """get_x_shape"""
        return self._x.shape

    def get_y_shape(self):
        """get_y_shape"""
        return self._y.shape

cmp_train = CMPModeIChamber4Dataset(X_train, y_train)
cmp_test = CMPModeIChamber4Dataset(X_test, y_test)

# ## 超参数

# Training parameters
BATCH_SIZE = 100
NUM_WORKERS = 0
LR = 0.001  # 2e-4
EPOCHS = 400  # 可以修改

# Model parameters
d_model = 8  # 64 # Lattent dim
q = 4  # Query size
v = 4  # Value size
h = 4  # Number of heads
N = 2  # 4 # Number of encoder and decoder to stack
attention_size = 6  # 12 # Attention window size # 可以修改
dropout = 0.2  # Dropout rate
pe = 'regular'  # None # Positional encoding(pe='original' and pe_period=None，或者pe='regular' and pe_period=int)
pe_period = 24
chunk_mode = None  # None

d_input = 19  # From dataset 是数据集的feature个数
d_output = 1  # From dataset
K = max_batch_length  # From dataset 是采样点个数，也是sequence_length

# Config
# sns.set() # 设置画图的配色
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # 此时3号GPU是空闲
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

dataloader_train = DataLoader(cmp_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=False
                             )
dataloader_test = DataLoader(cmp_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS
                            )


# 训练, 10 times
y_test_pre_list = []
mse_list, rmse_list, mae_list, r2_list = [], [], [], []
N_iter = 10
for i_iter in range(0, N_iter):
    print('*******************')
    print(f'Iter {i_iter} Start:')

    # Define Transformer Model
    net = Transformer(d_input, d_model, d_output, K, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe, pe_period=pe_period).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss(reduction='mean')

    model_save_path = f'../model/model_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth'
    test_loss_best = np.inf
    # Prepare loss history
    hist_loss = np.zeros(EPOCHS)
    hist_loss_test = np.zeros(EPOCHS)
    pred_test = []
    epoch_best = 0
    # training and test
    for idx_epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(dataloader_train.dataset), desc=f"[Epoch {idx_epoch+1:1d}/{EPOCHS}]") as pbar:
            for idx_batch, (x, y) in enumerate(dataloader_train):
                optimizer.zero_grad()

                # Propagate input
                netout = net(x.to(device))

                # Comupte loss
                loss = loss_function(y.to(device), netout)

                # Backpropage loss
                loss.backward()

                # Update weights
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
                pbar.update(x.shape[0])

            train_loss = running_loss/len(dataloader_train)
            test_loss, predictions = compute_loss(net, dataloader_test, loss_function, device)
            test_loss = test_loss.item()
            pred_test.append(predictions)
            pbar.set_postfix({'train loss': train_loss, 'test loss': test_loss})

            hist_loss[idx_epoch] = train_loss
            hist_loss_test[idx_epoch] = test_loss

        if test_loss < test_loss_best:
            test_loss_best = test_loss
            epoch_best = idx_epoch
        print('test_loss_best:', test_loss_best, 'epoch_best:', epoch_best)

    # Compute metrics
    y_test_pre = pred_test[epoch_best]  # choose best predicted y_test
    y_test_pre_list.append(y_test_pre)
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pre))
    mae = mean_absolute_error(y_test, y_test_pre)
    r2 = r2_score(y_test, y_test_pre)

    mse_list.append(mse);rmse_list.append(rmse);mae_list.append(mae);r2_list.append(r2)

# save mean predicted results
y_test_pre_mean = np.mean(y_test_pre_list, axis=0)
mse_mean, rmse_mean, mae_mean, r2_mean = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(r2_list)
print(f'Mean results of {N_iter} times iteration:')
print(f'Mean mse of {N_iter} times iteration:', mse_mean)
print(f'Mean rmse of {N_iter} times iteration:', rmse_mean)
print(f'Mean mae of {N_iter} times iteration:', mae)
print(f'Mean r2 of {N_iter} times iteration:', r2_mean)

dict_result = {}
dict_result = {'y_test_pre': y_test_pre_mean, 'mse': mse_mean, 'rmse': rmse_mean, 'mae': mae_mean, 'r2': r2_mean}
pickle.dump(dict_result, open(f'../results_save/results_tf.pkl', 'wb'))

# save last model info
pickle.dump(hist_loss, open('../results_save/train_loss_tf.pkl', 'wb'))
pickle.dump(hist_loss_test, open('../results_save/test_loss_tf.pkl', 'wb'))
torch.save(net, '../results_save/net.pth')

