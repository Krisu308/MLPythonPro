from pandas import *
from sklearn.preprocessing import MaxAbsScaler
import seaborn as sb
import matplotlib.pyplot as plt;
d=read_csv("D:\\2189\\BCAresult.csv")
print(d.head())
data=get_dummies(d["name"])
print(data.head())
scaler = MaxAbsScaler() 
maxabs_scaled_data = scaler.fit_transform(data) 
print(maxabs_scaled_data) 
