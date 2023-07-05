import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GTrend(nn.Module):
    def __init__(self,window,hidden_dim,num_layers,dropout,bidirectional,is_cuda,steps) -> None:
        super(GTrend,self).__init__()
        self.seq_len = window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_dir = 2 if bidirectional else 1
        self.is_cuda = is_cuda
        self.rnn_like = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)
        self.hidden2predict = nn.Linear(self.hidden_dim*self.num_dir,steps,bias=False)
    
    def forward(self,x):
        x = x.permute(1,0,2)
        if self.is_cuda:
            h_0 = Variable(torch.zeros(self.num_layers*self.num_dir, x.shape[1], self.hidden_dim).cuda())
        else:
            h_0 = Variable(torch.zeros(self.num_layers*self.num_dir, x.shape[1], self.hidden_dim))

        x,_ = self.rnn_like(x, h_0)
        x = x.permute(1,0,2)
        x = self.hidden2predict(x[:,-1,:])

        return x

class LTrend(nn.Module):
    def __init__(self,window,hidden_dim,num_layers,dropout,bidirectional,is_cuda,steps) -> None:
        super(LTrend,self).__init__()
        self.seq_len = window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_dir = 2 if bidirectional else 1
        self.is_cuda = is_cuda
        self.rnn_like = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)
        self.hidden2predict = nn.Linear(self.hidden_dim*self.num_dir,steps,bias=False)
    
    def forward(self,x):
        x = x.permute(1,0,2)
        if self.is_cuda:
            h_0 = Variable(torch.zeros(self.num_layers*self.num_dir, x.shape[1], self.hidden_dim).cuda())
            c_0 = Variable(torch.zeros(self.num_layers*self.num_dir, x.shape[1], self.hidden_dim).cuda())
        else:
            h_0 = Variable(torch.zeros(self.num_layers*self.num_dir, x.shape[1], self.hidden_dim))
            c_0 = Variable(torch.zeros(self.num_layers*self.num_dir, x.shape[1], self.hidden_dim))

        x,_ = self.rnn_like(x, (h_0, c_0))
        x = x.permute(1,0,2)
        x = self.hidden2predict(x[:,-1,:])

        return x

