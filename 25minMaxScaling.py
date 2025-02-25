from pandas import *
from sklearn.preprocessing import MinMaxScaler
d=read_csv("D:\\2189\\BCAresult.csv")
print(d.head())
data=get_dummies(d["name"])
print(data.head())
scaler = MinMaxScaler() 
scaled_data = scaler.fit_transform(data) 
print(scaled_data) 
