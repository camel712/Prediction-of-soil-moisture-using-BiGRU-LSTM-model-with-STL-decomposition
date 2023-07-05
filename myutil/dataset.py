import torch
import torch.utils.data as Data


class FenLiangSteps(Data.Dataset):
    def __init__(self,x_data,y_data,window,begin_index,channel,steps):
        self.window = window
        self.x_data = x_data
        self.y_data = y_data
        self.begin_index = begin_index
        self.channel = channel
        self.steps = steps
    def __len__(self):
        return len(self.x_data)-self.begin_index-self.steps
    def __getitem__(self,index):
        return torch.tensor(self.x_data[index+self.begin_index-self.window:index+self.begin_index,self.channel]).unsqueeze(-1),\
               torch.tensor(self.x_data[index+self.begin_index:index+self.begin_index+self.steps,self.channel])