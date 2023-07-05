import numpy as np
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from myutil.mydtw import my_dtw_tdi

cuda = torch.cuda.is_available()

device ="cuda" if torch.cuda.is_available() else "cpu"
def get_metric(predict_list,real_list):
    predict = np.vstack(predict_list)
    real = np.vstack(real_list)
    try:
        mse = mean_squared_error(real, predict)
        rmse = np.sqrt(mean_squared_error(real, predict))
        mae = mean_absolute_error(real, predict)
        r2 = r2_score(real,predict)
    except Exception as e:
        print(e)
        print(real)
        print("------------")
        print(predict)
    else:
        return mse,rmse,mae,r2

def train(dataloader, model, loss_fn, optimizer):
    model.train() 
    accum_step = 1
    pred_list = []
    real_list = []
    for batch, (X, y) in enumerate(dataloader): 

        X, y = X.type(torch.float32).to(device), y.type(torch.float32).to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward() 
        if (batch+1) % accum_step == 0:
            optimizer.step()                    
            optimizer.zero_grad()               
        
        pred_list.append(pred.detach().cpu().numpy())
        real_list.append(y.detach().cpu().numpy())

    print(f"train_set loss:{loss.item():>10f}")
    mse,rmse,mae,r2 = get_metric(pred_list,real_list)
    print(f"mse:{mse:>11f}  rmse:{rmse:>11f}  mae:{mae:>11f}  r2:{r2:>11f}")
    return mse,rmse,mae,r2

def test(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    pred_list = []
    real_list = []
    loss_dtw,loss_tdi = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.type(torch.float32).to(device), y.type(torch.float32).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            pred_cpu = pred.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            pred_list.append(pred_cpu)
            real_list.append(y_cpu)
            
            output_len = y.shape[1]
            if output_len>2:
                sim,tdi = my_dtw_tdi(y_cpu,pred_cpu)
                loss_dtw += sim
                loss_tdi += tdi
            else:
                loss_dtw,loss_tdi = 999,999
    test_loss /= num_batches
    dtw = loss_dtw /num_batches
    tdi = loss_tdi /num_batches

    print(f"Test Set: Avg loss: {test_loss:>10f}")
    mse,rmse,mae,r2 = get_metric(pred_list,real_list)
    r2 = 1-(1-r2)*(7933-1)/(7933-3-1)
    print(f"mse:{mse:>11f}  rmse:{rmse:>11f}  mae:{mae:>11f}  r2:{r2:>11f}")
    print(f"dtw:{dtw:>11f}  tdi:{tdi:>11f}")
    return mse,rmse,mae,r2,dtw,tdi,pred_list,real_list