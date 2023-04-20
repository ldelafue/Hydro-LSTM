# -*- coding: utf-8 -*-
"""
By using the Code or the HydroLSTM representation in your publication(s), you agree to cite:

De la Fuente, L. A., Ehsani, M. R., Gupta, H. V., and Condon, L. E.: 
Towards Interpretable LSTM-based Modelling of Hydrological Systems,
EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-666, 2023.

"""
import torch
import torch.nn as nn

#%%

class Model_lstm(nn.Module):
    def __init__(self, input_size: int, lag: int, state_size: int, dropout: float):
        super(Model_lstm, self).__init__()
        self.input_size = input_size
        self.sequence = lag #new
        self.state_size = state_size
        self.dropout = nn.Dropout(p=dropout)
        self.h_t = 0  # initialization 
        self.c_t = 0  # initialization 
        self.lstm = LSTM(self.input_size, self.state_size, self.sequence) #sequence is new
        self.regression = nn.Linear(state_size, 1, bias=True)
        #self.epoch = 0


    def forward(self, x):
        
        h_0 = x.data.new(1, self.state_size).zero_()
        c_0 = x.data.new(1, self.state_size).zero_()

        self.h_t, self.c_t = self.lstm(x, h_0, c_0)

        if self.state_size != 1:
            self.h_t_dropout = self.dropout(self.h_t)
            q_t = self.regression(self.h_t_dropout)

        else:
            q_t = self.regression(self.h_t)

        q_t = q_t[1:]
        return q_t, self.h_t, self.c_t
    
    
#%%

class LSTM(nn.Module):
    def __init__(self, input_size: int, state_size: int, sequence: int):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.sequence = sequence -1
        
        # create tensors parameters

        if self.state_size == 1:
            self.weight_input = nn.Parameter(torch.FloatTensor(4, self.input_size))  # (#input x 4 * #state variables)
        else:
            self.weight_input = nn.Parameter(torch.FloatTensor(self.state_size, 4, self.input_size))  # (#input x 4 * #state variables)

        # (#state_size x 4 * #state variables)
        self.weight_recur = nn.Parameter(torch.FloatTensor(4, self.state_size))
        self.bias = nn.Parameter(torch.FloatTensor(4, self.state_size))  # (4 * #state variables)

        # Initialize parameters for LSTM
        nn.init.xavier_uniform_(self.weight_input.data, gain=1.0)
        nn.init.xavier_uniform_(self.weight_recur.data, gain=1.0)
        nn.init.xavier_uniform_(self.bias.data, gain=1.0)


    def forward(self, x: torch.Tensor, h_0_input, c_0_input):

        batch_size = x.size(0)  # checking the batch size

        x_transp = torch.transpose(x, 0, 1)
        #print(x_transp.size())
        h_n = h_0_input
        c_n = c_0_input


        for it in range(batch_size):
            h_0 = torch.diagflat(h_0_input)
            c_0 = torch.diagflat(c_0_input)
            
            for seq in range(self.sequence + 1):
                #print(self.sequence)
                x_t = x_transp.data[(self.sequence - seq, self.sequence*2 + 1 - seq), it]
                x_t = x_t.resize(self.input_size, 1)
    
                if self.state_size == 1:
                    gates = (torch.addmm(self.bias, self.weight_recur, h_0) + torch.mm(self.weight_input, x_t))  # calculate gates
                else:
                    x_t = x_t.unsqueeze(dim=0)
                    x_t = x_t.repeat(self.state_size, 1, 1)
                    sum1 = torch.bmm(self.weight_input, x_t)
                    sum1 = sum1.squeeze()
                    sum1 = torch.transpose(sum1, 0, 1)
                    
                    # calculate gates
                    gates = (torch.addmm(self.bias, self.weight_recur, h_0) + sum1)
                f, i, o, g = gates.chunk(4, dim=0)
                f = torch.sigmoid(f)  # calculate activation [0,1]
                i = torch.sigmoid(i)  # calculate activation [0,1]
                o = torch.sigmoid(o)  # calculate activation [0,1]
                g = torch.tanh(g)  # calculate activation [-1,1]
    
                if self.state_size == 1:
                    c_1 = f * c_0 + i * g
    
                else:
                    c_1 = torch.mm(f, c_0) + torch.mul(i, g)
                    c_1 = torch.diagflat(c_1)
    
                h_1 = torch.mul(o, torch.tanh(c_1))  # linear reservoir
                h_0 = h_1
                c_0 = c_1


            # store output and state varaible in a list
            h_1 = torch.diag(h_1, 0)
            c_1 = torch.diag(c_1, 0)
            h_1 = h_1.unsqueeze(dim=0)
            c_1 = c_1.unsqueeze(dim=0)
            h_n = torch.cat((h_n, h_1), 0)
            c_n = torch.cat((c_n, c_1), 0)


        return h_n, c_n
