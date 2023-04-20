# -*- coding: utf-8 -*-
"""
By using the Code or the HydroLSTM representation in your publication(s), you agree to cite:

De la Fuente, L. A., Ehsani, M. R., Gupta, H. V., and Condon, L. E.: 
Towards Interpretable LSTM-based Modelling of Hydrological Systems,
EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-666, 2023.

"""
#%% Libraries
import argparse
#from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import *
from datetime import timedelta
from datetime import date
import random
import pickle
import pandas as pd
import numpy as np

#%% Functions
from Hydro_LSTM import *
from LSTM import *
from utils import *

#%% reading parameter
parser = argparse.ArgumentParser(description = "*** This code is calling all the other code files. These are the parameterization used. ***")


#parser.add_argument('--country', choices=["US", "CL"]) # Only US catchments is implemented here
parser.add_argument('--code', type=int, help='Catchment ID from CAMELS dataset') #ID 1000000 runs a unique model over the 10 catchment, do not implemented here
parser.add_argument('--cells', type=int, help='Number of state cells (in paralel) used in one hidden layer')
parser.add_argument('--memory', type=int, help='Number of days used in the sequence length [1,256]')
parser.add_argument('--epochs', type=int, default=512, help='Times the model is trained using the same dataset, Default=512')
#parser.add_argument('--patience', type=int, default=512)
parser.add_argument('--learning_rate', default=1e-4, help='Tuning parameter in an optimization algorithm that determines how fast we update the weights. Default=1e-4')
parser.add_argument('--processor', default="cpu", help='Where the model will be ran. Default="cpu"')
parser.add_argument('--model', choices=["LSTM", "HYDRO"], help='Model choises. "LSTM" or "HYDRO"')
#parser.add_argument('--normalization', choices=["Global", "Local"]) # Only valid for a unique model, do not implemented here
#parser.add_argument('--attribute', type=int, default=0)  # only usesful for with a unique model
cfg = vars(parser.parse_args())


#country = cfg["country"]
code = cfg["code"]
cells = cfg["cells"]
memory = cfg["memory"]
learning_rate = cfg["learning_rate"]
epochs = cfg["epochs"]
patience= epochs
processor = cfg["processor"]
model_option = cfg["model"]
#normalization = cfg["normalization"]
#n_attibutes = cfg["attribute"]

country='US'
n_variables = 2 #ex. PP and PET = 2 variables
n_attibutes = 0 #1:Aridity ----It only works with Global normalization
dropout = 0
#model_option = "HYDRO" #"LSTM
normalization = "local" #"Local" or "Global" ONLY FOR CODE=1000000


#Recent rainfall-dominant (West)
#code = 11523200 #TRINITY R AB COFFEE C NR TRINITY CTR CA  01/1980
#code = 11473900 # MF EEL R NR DOS RIOS CA 01/1980

#Snowmelt-dominant
#code = 9223000 #HAMS FORK BELOW POLE CREEK, NEAR FRONTIER, WY  01/1980
#code = 9035900 #SOUTH FORK OF WILLIAMS FORK NEAR LEAL, CO. 01/1980

#Mixed
#code = 6847900 #PRAIRIE DOG C AB KEITH SEBELIUS LAKE, KS 01/1980
#code = 6353000 #CEDAR CREEK NR RALEIGH, ND 01/1980

#Historical rainfall-dominant
#code = 2472000 #LEAF RIVER NR COLLINS, MS 01/1980
#code = 5362000 #JUMP RIVER AT SHELDON, WI 01/1980

#Recent rainfall-dominant (East)
#code = 3173000 #WALKER CREEK AT BANE, VA 01/1980
#code = 1539000 #Fishing Creek near Bloomsburg, PA 01/1980

#All
#code = 1000000


#%%

if code== 1000000:
    predictions = np.empty([15542,21])
    results = pd.DataFrame(columns=['lag', 'batch', 'cell', 'RMSE', 'MAE', 'R2', 'CC', 'Bias', 'KGE'])
    results_catchment = pd.DataFrame(columns=['code','lag', 'batch', 'cell', 'RMSE', 'MAE', 'R2', 'CC', 'Bias', 'KGE'])
else:
    predictions = np.empty([10321,21])
    state_results = np.empty([10321,20])
    results = pd.DataFrame(columns=['lag', 'batch', 'cell', 'RMSE', 'MAE', 'R2', 'CC', 'Bias', 'KGE'])


lag_values = [1]
lag_values = [memory*s for s in lag_values] #[2,4,8,16,32,64,128,256]
batch_size_values = [8]
state_size_values = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
state_size_values = [cells*s for s in state_size_values] #1,2,3,4,8,16
model_summary = []
i=0
for lag in lag_values:
     lag = lag + 1
     for batch_size in batch_size_values:
         for state_size in state_size_values:

            x_max = []
            x_min = []
            x_mean = []
            y_max = []
            y_min = []
            y_mean = []

            warm_up = lag
            input_size= n_variables*lag + n_attibutes


            if processor == "cpu":
                DEVICE = torch.device("cpu")
            else:
                DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(DEVICE)

            if code == 1000000:
                code_list =    [11523200,11473900,9223000,9035900,6847900,6353000,2472000,5362000,3173000,1539000]
                code_aridity = {11523200:0.873611,
                                11473900:0.765788,
                                9223000:1.57739,
                                9035900:1.43871,
                                6847900:2.09800,
                                6353000:2.68061,
                                2472000:0.76175,
                                5362000:0.88262,
                                3173000:0.87104,
                                1539000:0.68940}

                random.shuffle(code_list)
                z=1
                for code_i in code_list:
                    PP_i, PET_i, Q_i = load_data(code_i, country, warm_up)

                    ini_training = PP_i.index[0] + timedelta(days=275)
                    initial = ini_training - timedelta(days=lag)

                    n=len(PP_i[initial:])

                    end_training = 7303 #int(0.7*n) #8000
                    end_valid = end_training + 1462 #int(0.85*n) #9000

                    PP_train_i = PP_i.loc[initial:ini_training + timedelta(days=end_training),:]
                    PET_train_i = PET_i.loc[initial:ini_training + timedelta(days=end_training),:]
                    Q_train_i = Q_i.loc[initial:ini_training + timedelta(days=end_training),:]

                    PP_test_i = PP_i.loc[ini_training + timedelta(days=end_valid) - timedelta(days=lag):,:]
                    PET_test_i = PET_i.loc[ini_training + timedelta(days=end_valid)- timedelta(days=lag):,:]
                    Q_test_i = Q_i.loc[ini_training + timedelta(days=end_valid)- timedelta(days=lag):,:]

                    PP_valid_i = PP_i.loc[initial:ini_training + timedelta(days=end_valid - 1),:]
                    PET_valid_i = PET_i.loc[initial:ini_training + timedelta(days=end_valid - 1),:]
                    Q_valid_i = Q_i.loc[initial:ini_training + timedelta(days=end_valid - 1),:]

                    if z==1:
                        PP, PET, Q = load_data(code_i, country, warm_up)

                        PP_train = PP_train_i.copy()
                        PET_train = PET_train_i.copy()
                        Q_train = Q_train_i.copy()

                        PP_test = PP_test_i.copy()
                        PET_test = PET_test_i.copy()
                        Q_test = Q_test_i.copy()

                        PP_valid = PP_valid_i.copy()
                        PET_valid = PET_valid_i.copy()
                        Q_valid = Q_valid_i.copy()
                        ds = torch_dataset(PP_train_i,PET_train_i,Q_train_i, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True) #create a class with x and y as torch dataset
                        if  normalization == "Global":
                            ds.y = ds.y*(ds.y_max - ds.y_min) + ds.y_mean
                            ds.x = ds.x*(ds.x_max - ds.x_min) + ds.x_mean
                            if n_attibutes >= 1:
                                arid_tensor = code_aridity[code_i]*torch.ones((ds.x.shape[0],1))
                                ds.x = torch.cat((ds.x,arid_tensor), 1)
                            if n_attibutes == 2:
                                elev_tensor = code_Elev[code_i]*torch.ones((ds.x.shape[0],1))
                                ds.x = torch.cat((ds.x,elev_tensor), 1)

                            ds.x = ds.x.float()


                        else:
                            x_max = ds.x_max
                            x_min = ds.x_min
                            x_mean = ds.x_mean
                            y_max = ds.y_max
                            y_min = ds.y_min
                            y_mean = ds.y_mean


                        ds_valid = torch_dataset(PP_valid_i,PET_valid_i, Q_valid_i, lag, ini_training, ds.x_max, ds.x_min, ds.x_mean, ds.y_max, ds.y_min, ds.y_mean, istrain=False)
                        ds_valid.x = ds_valid.x[end_training + 1:,:]
                        ds_valid.y = ds_valid.y[end_training + 1: ]
                        ds_valid.num_samples = len(ds_valid.y)
                        if  normalization == "Global":
                            ds_valid.y = ds_valid.y*(ds.y_max - ds.y_min) + ds.y_mean
                            ds_valid.x = ds_valid.x*(ds.x_max - ds.x_min) + ds.x_mean
                            if n_attibutes >= 1:
                                arid_tensor = code_aridity[code_i]*torch.ones((ds_valid.x.shape[0],1))
                                ds_valid.x = torch.cat((ds_valid.x,arid_tensor), 1)
                            if n_attibutes == 2:
                                elev_tensor = code_Elev[code_i]*torch.ones((ds_valid.x.shape[0],1))
                                ds_valid.x = torch.cat((ds_valid.x,elev_tensor), 1)


                            ds_valid.x = ds_valid.x.float()

                        ds_full = torch_dataset(PP_test_i,PET_test_i,Q_test_i, lag, ini_training + timedelta(days=end_valid) , ds.x_max, ds.x_min, ds.x_mean, ds.y_max, ds.y_min, ds.y_mean, istrain=False) #IT IS NOT FULL
                        if  normalization == "Global":
                            ds_full.y = ds_full.y*(ds.y_max - ds.y_min) + ds.y_mean
                            ds_full.x = ds_full.x*(ds.x_max - ds.x_min) + ds.x_mean
                            if n_attibutes >= 1:
                                arid_tensor = code_aridity[code_i]*torch.ones((ds_full.x.shape[0],1))
                                ds_full.x = torch.cat((ds_full.x,arid_tensor), 1)
                            if n_attibutes == 2:
                                elev_tensor = code_Elev[code_i]*torch.ones((ds_full.x.shape[0],1))
                                ds_full.x = torch.cat((ds_full.x,elev_tensor), 1)
                            ds_full.x = ds_full.x.float()

                    else:
                        PP = PP.append(PP_i, ignore_index=True)
                        PET = PET.append(PET_i, ignore_index=True)
                        Q = Q.append(Q_i, ignore_index=True)

                        PP_train = PP_train.append(PP_train_i, ignore_index=True)
                        PET_train = PET_train.append(PET_train_i, ignore_index=True)
                        Q_train = Q_train.append(Q_train_i, ignore_index=True)

                        PP_test = PP_test.append(PP_test_i, ignore_index=True)
                        PET_test = PET_test.append(PET_test_i, ignore_index=True)
                        Q_test = Q_test.append(Q_test_i, ignore_index=True)

                        PP_valid = PP_valid.append(PP_valid_i, ignore_index=True)
                        PET_valid = PET_valid.append(PET_valid_i, ignore_index=True)
                        Q_valid = Q_valid.append(Q_valid_i, ignore_index=True)

                        ds_i = torch_dataset(PP_train_i,PET_train_i,Q_train_i, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True) #create a class with x and y as torch dataset
                        if  normalization == "Global":
                            ds_i.y = ds_i.y*(ds_i.y_max - ds_i.y_min) + ds_i.y_mean
                            ds_i.x = ds_i.x*(ds_i.x_max - ds_i.x_min) + ds_i.x_mean
                            if n_attibutes >= 1:
                                arid_tensor = code_aridity[code_i]*torch.ones((ds_i.x.shape[0],1))
                                ds_i.x = torch.cat((ds_i.x,arid_tensor), 1)
                            if n_attibutes == 2:
                                elev_tensor = code_Elev[code_i]*torch.ones((ds_i.x.shape[0],1))
                                ds_i.x = torch.cat((ds_i.x,elev_tensor), 1)


                        else:
                            x_max = np.append(x_max,ds_i.x_max)
                            x_min = np.append(x_min,ds_i.x_min)
                            x_mean = np.append(x_mean,ds_i.x_mean)
                            y_max= np.append(y_max,ds_i.y_max)
                            y_min = np.append(y_min,ds_i.y_min)
                            y_mean = np.append(y_mean,ds_i.y_mean)

                        ds.x = torch.cat((ds.x,ds_i.x),0)
                        ds.y = torch.cat((ds.y,ds_i.y),0)
                        ds.num_samples = len(ds.y)
                        ds.x = ds.x.float()




                        ds_valid_i = torch_dataset(PP_valid_i,PET_valid_i, Q_valid_i, lag, ini_training, ds_i.x_max, ds_i.x_min, ds_i.x_mean, ds_i.y_max, ds_i.y_min, ds_i.y_mean, istrain=False)
                        ds_valid_i.x = ds_valid_i.x[end_training + 1:,:]
                        ds_valid_i.y = ds_valid_i.y[end_training + 1: ]
                        ds_valid_i.num_samples = len(ds_valid_i.y)
                        if  normalization == "Global":
                            ds_valid_i.y = ds_valid_i.y*(ds_i.y_max - ds_i.y_min) + ds_i.y_mean
                            ds_valid_i.x = ds_valid_i.x*(ds_i.x_max - ds_i.x_min) + ds_i.x_mean
                            if n_attibutes >= 1:
                                arid_tensor = code_aridity[code_i]*torch.ones((ds_valid_i.x.shape[0],1))
                                ds_valid_i.x = torch.cat((ds_valid_i.x,arid_tensor), 1)
                            if n_attibutes == 2:
                                elev_tensor = code_Elev[code_i]*torch.ones((ds_valid_i.x.shape[0],1))
                                ds_valid_i.x = torch.cat((ds_valid_i.x,elev_tensor), 1)

                        ds_valid.x = torch.cat((ds_valid.x,ds_valid_i.x),0)
                        ds_valid.y = torch.cat((ds_valid.y,ds_valid_i.y),0)
                        ds_valid.x = ds_valid.x.float()


                        ds_full_i = torch_dataset(PP_test_i,PET_test_i,Q_test_i, lag, ini_training + timedelta(days=end_valid) , ds_i.x_max, ds_i.x_min, ds_i.x_mean, ds_i.y_max, ds_i.y_min, ds_i.y_mean, istrain=False) #IT IS NOT FULL
                        if  normalization == "Global":
                            ds_full_i.y = ds_full_i.y*(ds_i.y_max - ds_i.y_min) + ds_i.y_mean
                            ds_full_i.x = ds_full_i.x*(ds_i.x_max - ds_i.x_min) + ds_i.x_mean
                            if n_attibutes >= 1:
                                arid_tensor = code_aridity[code_i]*torch.ones((ds_full_i.x.shape[0],1))
                                ds_full_i.x = torch.cat((ds_full_i.x,arid_tensor), 1)
                            if n_attibutes == 2:
                                elev_tensor = code_Elev[code_i]*torch.ones((ds_full_i.x.shape[0],1))
                                ds_full_i.x = torch.cat((ds_full_i.x,elev_tensor), 1)

                        ds_full.x = torch.cat((ds_full.x,ds_full_i.x),0)
                        ds_full.y = torch.cat((ds_full.y,ds_full_i.y),0)
                        ds_full.x = ds_full.x.float()


                    z = z+1

                #ds = torch_dataset(PP_train,PET_train,Q_train, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True) #create a class with x and y as torch dataset

                if  normalization == "Global":
                    ds.x_max = ds.x.max(axis=0)[0]
                    ds.x_min = ds.x.min(axis=0)[0]
                    ds.x_mean = ds.x.mean(axis=0) #[-1,1]

                    ds.y_max = ds.y.max()
                    ds.y_min = ds.y.min()
                    ds.y_mean = ds.y.mean() #[-1,1]

                    ds.y = (ds.y - ds.y_mean)/(ds.y_max - ds.y_min)
                    ds.x = torch.add(ds.x, torch.mul(ds.x_mean,-1))
                    range_ds = ds.x_max - ds.x_min
                    ds.x = torch.div(ds.x, range_ds)

                    ds_valid.y = (ds_valid.y - ds.y_mean)/(ds.y_max - ds.y_min)
                    ds_valid.x = torch.add(ds_valid.x, torch.mul(ds.x_mean,-1))
                    ds_valid.x = torch.div(ds_valid.x, range_ds)


                    x_max = ds.x_max.numpy()
                    x_min = ds.x_min.numpy()
                    x_mean = ds.x_mean.numpy()
                    y_max = ds.y_max.numpy()
                    y_min = ds.y_min.numpy()
                    y_mean = ds.y_mean.numpy()

                    ds_full.y = (ds_full.y - ds.y_mean)/(ds.y_max - ds.y_min)
                    ds_full.x = torch.add(ds_full.x, torch.mul(ds.x_mean,-1))
                    ds_full.x = torch.div(ds_full.x, range_ds)

                ds_full.num_samples = len(ds_full.y)
                ds_valid.num_samples = len(ds_valid.y)




            else:
                PP, PET, Q = load_data(code, country, warm_up)
                #print('PP:',PP.shape)
                #print('PET:',PET.shape)
                #print('Q:',Q.shape)

                ini_training = PP.index[0] + timedelta(days=275)
                initial = ini_training - timedelta(days=lag)

                n=len(PP[initial:])

                end_training = 7303 #int(0.7*n) #8000
                end_valid = end_training + 1462 #int(0.85*n) #9000

                PP_train = PP.loc[initial:ini_training + timedelta(days=end_training),:]
                PET_train = PET.loc[initial:ini_training + timedelta(days=end_training),:]
                Q_train = Q.loc[initial:ini_training + timedelta(days=end_training),:]

                PP_test = PP.loc[ini_training + timedelta(days=end_valid):,:]
                PET_test = PET.loc[ini_training + timedelta(days=end_valid):,:]
                Q_test = Q.loc[ini_training + timedelta(days=end_valid):,:]

                PP_valid = PP.loc[initial:ini_training + timedelta(days=end_valid - 1),:]
                PET_valid = PET.loc[initial:ini_training + timedelta(days=end_valid - 1),:]
                Q_valid = Q.loc[initial:ini_training + timedelta(days=end_valid - 1),:]



                ds = torch_dataset(PP_train,PET_train,Q_train, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True) #create a class with x and y as torch dataset
                x_max = ds.x_max
                x_min = ds.x_min
                x_mean = ds.x_mean
                y_max = ds.y_max
                y_min = ds.y_min
                y_mean = ds.y_mean

                ds_valid = torch_dataset(PP_valid,PET_valid, Q_valid, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=False)
                ds_valid.x = ds_valid.x[end_training + 1:,:]
                ds_valid.y = ds_valid.y[end_training + 1: ]
                ds_valid.num_samples = len(ds_valid.y)

                ### TESTING
                ds_full = torch_dataset(PP.loc[initial:,:],PET.loc[initial:,:],Q.loc[initial:,:], lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=False)
                sampler_test = SequentialSampler(ds_full)
                loader_test = DataLoader(ds_full, batch_size=batch_size, sampler=sampler_test, shuffle=False)


            if model_option == "HYDRO":
                sampler = SequentialSampler(ds)
                loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False) # create subset (batches) of the data
                sampler_valid = SequentialSampler(ds_valid)
                loader_valid = DataLoader(ds_valid, batch_size=batch_size, sampler=sampler_valid, shuffle=False) # create subset (batches) of the data
                model = Model_hydro_lstm(input_size, lag, state_size, dropout).to(DEVICE)
                sampler_test = SequentialSampler(ds_full)
                loader_test = DataLoader(ds_full, batch_size=batch_size, sampler=sampler_test, shuffle=False)

            else:
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True,num_workers=2) # create subset (batches) of the data
                loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True,num_workers=2) # create subset (batches) of the data
                model = Model_lstm(int(input_size/lag), lag, state_size, dropout).to(DEVICE)
                sampler_test = SequentialSampler(ds_full)
                loader_test = DataLoader(ds_full, batch_size=batch_size, sampler=sampler_test, shuffle=False,num_workers=2)


            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            loss_func = nn.SmoothL1Loss() #nn.L1Loss()

            learning_rates = {200: learning_rate, 250: learning_rate} # in case we want to change the learning rate for different epoch

            valid_losses = [] # to track the validation loss as the model trains
            model_list = []

            for epoch in range(1, epochs + 1):

                # set new learning rate
                if epoch in learning_rates.keys():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rates[epoch]
                stopping, model_list, valid_losses = train_epoch(model, optimizer, loss_func, loader, epoch, loader_valid, patience, model_list, valid_losses, DEVICE)
                if stopping:
                    break




            iteration = 1
            for data in loader_test:

                x_epoch, y_epoch = data

                x_epoch, y_epoch = x_epoch.to(DEVICE), y_epoch.to(DEVICE)
                y_epoch = y_epoch.resize(len(y_epoch),1)

                pred_epoch = model(x_epoch)[0]
                c_epoch = model(x_epoch)[2]
                c_epoch = c_epoch.cpu().detach().numpy()

                pred_epoch = pred_epoch.cpu().detach().numpy()
                y_epoch = y_epoch.cpu().detach().numpy() #
                if iteration == 1:
                    q_sim = pred_epoch
                    q_obs = y_epoch
                    if state_size == 1:
                        state_value = c_epoch[1:]
                else:
                    q_sim = np.concatenate((q_sim, pred_epoch))
                    q_obs = np.concatenate((q_obs, y_epoch))
                    if state_size == 1:
                        state_value = np.concatenate((state_value, c_epoch[1:]))
                iteration = iteration + 1


            # q_sim = model(ds_full.x.to(DEVICE))[0]
            # q_obs = ds_full.y.to(DEVICE)
            # state_var = model.c_t.data
            # h_var = model.h_t.data


            if code == 1000000:
                l_test = int(len(q_sim)/len(code_list))
                if  normalization == "Global":
                    q_sim = q_sim*(y_max - y_min) + y_mean
                    q_obs = q_obs*(y_max - y_min) + y_mean
                else:
                    for ii in range(len(code_list)):
                        q_sim[ii*l_test:(ii+1)*l_test] = q_sim[ii*l_test:(ii+1)*l_test]*(y_max[ii] - y_min[ii]) + y_mean[ii]
                        q_obs[ii*l_test:(ii+1)*l_test] = q_obs[ii*l_test:(ii+1)*l_test]*(y_max[ii] - y_min[ii]) + y_mean[ii]

                q_sim = q_sim.flatten()
                q_obs = q_obs.flatten()

                RMSE = mean_squared_error(q_sim, q_obs)**0.5
                MAE = mean_absolute_error(q_sim, q_obs)
                R2 = r2_score(q_sim, q_obs)
                BIAS = q_sim.sum() / q_obs.sum()
                CC = np.corrcoef([q_sim, q_obs],rowvar=True)
                CC = CC[0,1]
                mean_s = q_sim.mean()
                mean_o = q_obs.mean()
                std_s = q_sim.std()
                std_o = q_obs.std()
                KGE = 1 - ((CC - 1) ** 2 + (std_s / std_o - 1) ** 2 + (mean_s / mean_o - 1) ** 2) ** 0.5

                results.at[i,:] = [lag-1, batch_size, state_size, RMSE, MAE, R2, CC, BIAS, KGE]
                print(results)

                for ii in range(len(code_list)):

                    RMSE_ii = mean_squared_error(q_sim[ii*l_test:(ii+1)*l_test], q_obs[ii*l_test:(ii+1)*l_test])**0.5
                    MAE_ii = mean_absolute_error(q_sim[ii*l_test:(ii+1)*l_test], q_obs[ii*l_test:(ii+1)*l_test])
                    R2_ii = r2_score(q_sim[ii*l_test:(ii+1)*l_test], q_obs[ii*l_test:(ii+1)*l_test])
                    BIAS_ii = q_sim[ii*l_test:(ii+1)*l_test].sum() / q_obs[ii*l_test:(ii+1)*l_test].sum()
                    CC_ii = np.corrcoef([q_sim[ii*l_test:(ii+1)*l_test], q_obs[ii*l_test:(ii+1)*l_test]],rowvar=True)
                    CC_ii = CC_ii[0,1]
                    mean_s_ii = q_sim[ii*l_test:(ii+1)*l_test].mean()
                    mean_o_ii = q_obs[ii*l_test:(ii+1)*l_test].mean()
                    std_s_ii = q_sim[ii*l_test:(ii+1)*l_test].std()
                    std_o_ii = q_obs[ii*l_test:(ii+1)*l_test].std()
                    KGE_ii = 1 - ((CC_ii - 1) ** 2 + (std_s_ii / std_o_ii - 1) ** 2 + (mean_s_ii / mean_o_ii - 1) ** 2) ** 0.5

                    results_catchment.at[i*len(code_list) + ii,:] = [code_list[ii],lag-1, batch_size, state_size, RMSE_ii, MAE_ii, R2_ii, CC_ii, BIAS_ii, KGE_ii]
                print(results_catchment)
                predictions[0,i+1] = lag-1
                predictions[1,i+1] = state_size
                predictions[2:,i+1] = q_sim
                predictions[2:,0] = q_obs

            else:

                q_sim = q_sim*(y_max - y_min) + y_mean
                q_obs = q_obs*(y_max - y_min) + y_mean

                q_sim = q_sim.flatten()
                q_obs = q_obs.flatten()
                state_value = state_value.flatten()


                RMSE = mean_squared_error(q_sim[end_valid+2:], q_obs[end_valid+2:])**0.5
                MAE = mean_absolute_error(q_sim[end_valid+2:], q_obs[end_valid+2:])
                R2 = r2_score(q_sim[end_valid+2:], q_obs[end_valid+2:])
                BIAS = q_sim[end_valid+2:].sum() / q_obs[end_valid+2:].sum()
                CC = np.corrcoef([q_sim[end_valid+2:], q_obs[end_valid+2:]],rowvar=True)
                CC = CC[0,1]
                mean_s = q_sim[end_valid+2:].mean()
                mean_o = q_obs[end_valid+2:].mean()
                std_s = q_sim[end_valid+2:].std()
                std_o = q_obs[end_valid+2:].std()
                KGE = 1 - ((CC - 1) ** 2 + (std_s / std_o - 1) ** 2 + (mean_s / mean_o - 1) ** 2) ** 0.5

                results.at[i,'lag'] = lag-1
                results.at[i,'batch'] = batch_size
                results.at[i,'cell'] = state_size
                results.at[i,'RMSE'] = RMSE
                results.at[i,'MAE'] = MAE
                results.at[i,'R2'] = R2
                results.at[i,'CC'] = CC
                results.at[i,'Bias'] = BIAS
                results.at[i,'KGE'] = KGE

                print(results)
                predictions[0,i+1] = lag-1
                predictions[1,i+1] = state_size
                predictions[2:,i+1] = q_sim
                predictions[2:,0] = q_obs
                state_results[0,i] = lag-1
                state_results[1,i] = state_size
                state_results[2:,i] = state_value


            i = i + 1
            model_summary.append(model)

if model_option == "HYDRO":
    name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_summary.csv'
    results.to_csv(name_file)
    name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_models.pkl'
    pickle.dump(model_summary, open(name_file, 'wb'))
    if code == 1000000:
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_summary_per_catchment.csv'
        results_catchment.to_csv(name_file)
    else:
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_predictions.csv'
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(name_file)
        if state_size == 1:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_state.csv'
        state_results = pd.DataFrame(state_results)
        state_results.to_csv(name_file)

else:
    name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_lstm_summary_elev.csv'
    results.to_csv(name_file)
    name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_lstm_models_elev.pkl'
    pickle.dump(model_summary, open(name_file, 'wb'))
    if code == 1000000:
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_lstm_summary_per_catchment_elev.csv'
        results_catchment.to_csv(name_file)
    else:
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_lstm_predictions.csv'
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(name_file)
