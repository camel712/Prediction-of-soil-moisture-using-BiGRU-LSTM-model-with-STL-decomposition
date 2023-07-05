import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

data = pd.read_csv("SM_AL02_5cm.csv")
data["time"] = pd.to_datetime(data[["year","month","day","hour"]])
data.set_index("time",inplace=True)
data.drop(["year","month","day","hour"],axis=1,inplace=True)

data["vmc"] = data["VMC"]*(2/(data["VMC"].max()-data["VMC"].min()))+1-(2*data["VMC"].max()/(data["VMC"].max()-data["VMC"].min()))

train = data.iloc[0:31720,:]


train_res = STL(train["vmc"],period=24*365,seasonal=7).fit()
train_dec_res = pd.DataFrame({"trend":train_res.trend,"season":train_res.seasonal,"resid":train_res.resid})
train_dec_res.to_csv("train_stl_res_2.csv",index=False)

data_res = STL(data["vmc"],period=24*365,seasonal=7).fit()
data_dec_res = pd.DataFrame({"trend":data_res.trend,"season":data_res.seasonal,"resid":data_res.resid})
data_dec_res.to_csv("data_stl_res_2.csv",index=False)