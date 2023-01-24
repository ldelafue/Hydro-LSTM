# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:14:04 2021
modified Jan 14
@author: Luis De la Fuente
"""

#from importing_MAC import * 
#from importing_notebook import * 
from importing import *
from datetime import timedelta
import torch
from tqdm import tqdm
import sys


#%%
def load_data(code, country, warm_up):
    PP, PET, Q = importing(code, country)
       
    Q['Q_obs'] = pd.to_numeric(Q['Q_obs'],errors = 'coerce')

    Q = Q[:PP.index[-1]]

    Q_nan = Q.dropna()
    if PP.index[0] + pd.DateOffset(days=warm_up) < Q_nan.index[0]:
        PP = PP[Q_nan.index[0] - pd.DateOffset(days=warm_up):]
        PET = PET[Q_nan.index[0] - pd.DateOffset(days=warm_up):]
        Q = Q[Q_nan.index[0]:]
    else:
        Q = Q[PP.index[0] + pd.DateOffset(days=warm_up):]
    return PP, PET, Q

#%%
class torch_dataset():
    
    def __init__(self, PP,PET,Q, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain):
              
        for i in range(1,lag):

            PP_name = 'PP_' + str(i)
            PP_copy = PP.copy()
            PP_copy[PP_name] = PP['PP'].shift(i)
            PP = PP_copy.copy()

            
            PET_name = 'PET_' + str(i)
            PET_copy = PET.copy()
            PET_copy[PET_name] = PET['PET'].shift(i)
            PET = PET_copy.copy()
                  
        X = pd.concat([PP, PET], axis=1)
        X = X.drop('basin', axis=1)
        X.at[:,'Q'] = Q.Q_obs #        X['Q'] = Q.Q_obs

        X = X.loc[ini_training - timedelta(days=1)  :,:]
        X = X.drop('Q', axis=1)

        x = X.values
        Q = Q.loc[X.index]
        y = Q.Q_obs.values
        
        if istrain:          
            self.x_max = x.max(axis=0)
            self.x_min = x.min(axis=0)
            self.x_mean = x.mean(axis=0) #[-1,1]
        
            self.y_max = y.max()
            self.y_min = y.min()
            self.y_mean = y.mean() #[-1,1]
        else:
            self.x_max = x_max
            self.x_min = x_min
            self.x_mean = x_mean
            self.y_max = y_max
            self.y_min = y_min
            self.y_mean = y_mean
                
        y = (y - self.y_mean)/(self.y_max - self.y_min)
        x = (x - self.x_mean)/(self.x_max - self.x_min) 
        
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.num_samples = self.x.shape[0]       
        
    def __len__(self):
        return self.num_samples   

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
    
#%%
def train_epoch(model, optimizer, loss_func, loader, epoch, loader_valid, patience, model_list, mean_valid_losses, DEVICE):
     
    
    stopping = False
    
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')
    # Iterate in batches over training set
    train_losses = []
    for data in pbar:
        
        optimizer.zero_grad()# delete old gradients
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        y = y.resize(len(y),1) #y.resize_(len(y),1) #
        
        model.epoch = epoch
        model.DEVICE = DEVICE
        predictions = model(x)[0]
        c_pred = model.c_t.data
                
        loss = loss_func(predictions, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step() # perform parameter update

        train_losses.append(loss.item())
        #pbar.set_postfix_str(f"Loss: {loss.item():5f}")
       
    total = sum(train_losses)
    length = len(train_losses)
    mean_loss = total/length
    print(f"Loss_train: {mean_loss:5f}")
    
    # Iterate in batches over valid set
    valid_losses = []

    for data in loader_valid:

        x_valid, y_valid = data
       
        x_valid, y_valid = x_valid.to(DEVICE), y_valid.to(DEVICE)
        y_valid = y_valid.resize(len(y_valid),1) 
        
        pred_valid = model(x_valid)[0]
                
        loss_valid = loss_func(pred_valid, y_valid)

        valid_losses.append(loss_valid.item())
        #print(valid_losses)


    total_valid = sum(valid_losses)
    length_valid = len(valid_losses)
    epoch_valid_loss = total_valid/length_valid
    mean_valid_losses.append(epoch_valid_loss)
    
    print(f"Loss_valid: {epoch_valid_loss:5f}")
    
    model_list.append(model)     

    # if epoch >= patience:
    #     if valid_losses[epoch-1] > valid_losses[epoch - patience]:
    #         model = model_list[valid_losses.index(min(valid_losses))]
    #         #model = model_list[epoch - patience]
    #         print("Early stopping")
    #         stopping = True

    if epoch == patience:
        index_best = mean_valid_losses.index(min(mean_valid_losses))
        model = model_list[index_best]
        print(f"Best model is selected in epoch: {int(index_best +1)}")
        stopping = True
    return stopping, model_list, mean_valid_losses
