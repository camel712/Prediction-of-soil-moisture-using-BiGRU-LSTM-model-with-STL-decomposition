import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from myutil.dataset import FenLiangSteps
from myutil.model_0 import GTrend,LTrend
from myutil.train import train,test
import sys

train_x = pd.read_csv("data/train_stl_res_2.csv")
train_label = pd.read_csv("data/train_res.csv")
valid_x = pd.read_csv("data/data_stl_res_2.csv")
valid_label = pd.read_csv("data/data_res.csv")

window = 24*365
step = int(sys.argv[2])
train_begin_index = window+1
valid_begin_index = 31720
device ="cuda" if torch.cuda.is_available() else "cpu"
batch_size = 200

#组合模型
t_model = GTrend(window=24*365,hidden_dim=32,num_layers=1,dropout=0,bidirectional=True,is_cuda=True,steps=step).to(device)
s_model = LTrend(window=24*365,hidden_dim=32,num_layers=1,dropout=0,bidirectional=False,is_cuda=True,steps=step).to(device)
r_model = LTrend(window=24*365,hidden_dim=32,num_layers=1,dropout=0,bidirectional=False,is_cuda=True,steps=step).to(device)
models = [t_model,s_model,r_model]
epochs_list = [1,1,1]
for i,model in enumerate(models):
    
    train_dataset = FenLiangSteps(x_data=train_x.to_numpy(),y_data=train_label["vmc"].to_numpy(),
                          window=window,begin_index=train_begin_index,channel=i,steps=step)
    valid_dataset = FenLiangSteps(x_data=valid_x.to_numpy(),y_data=valid_label["vmc"].to_numpy(),
                          window=window,begin_index=valid_begin_index,channel=i,steps=step)                          
    train_loader = Data.DataLoader(
        dataset=train_dataset,      
        batch_size=batch_size,      
        shuffle=True,               
        num_workers=12,             
        drop_last = False)

    test_loader = Data.DataLoader(
        dataset=valid_dataset,      
        batch_size=batch_size,      
        shuffle=True,               
        num_workers=12,             
        drop_last = False)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    train_metric_list = []
    test_metric_list = []
    
    path = sys.argv[1]+f"{i}/"
    epochs = epochs_list[i]
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        train_metric = train(train_loader, model, loss_fn, optimizer)
        test_metric = test(test_loader, model, loss_fn)
    
        train_metric_list.append(train_metric)
        test_metric_list.append(test_metric[0:6])
    
        #save_dict = {"model_param":model.state_dict(),"optim_param":optimizer.state_dict(),
        #                 "epoch":t,"train_metric_list":train_metric_list,"test_metric_list":test_metric_list}
        #torch.save(save_dict, f"{path}epoch{t}_rmse_{test_metric[1]:>5f}.pth")

    #save_dict = {"model_param":model.state_dict(),"optim_param":optimizer.state_dict(),
    #             "epoch":t,"train_metric_list":train_metric_list,"test_metric_list":test_metric_list}
    #torch.save(save_dict, f"{path}epoch{t}_rmse_{test_metric[1]:>5f}.pth")

