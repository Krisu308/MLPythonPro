import pandas as pd
from sklearn import preprocessing
data=pd.read_csv("D:\\2189\\BCAresult.csv")
print(data.head())
ohe=pd.get_dummies(data["name"])
print(ohe.head())
print("Before scaling Data:")
print("Mean is:\n",ohe.mean(axis=0))
print("After scaling Data:\n")
meanstd=preprocessing.scale(ohe)
print("Mean is:\n",meanstd.mean(axis=0))
print("Standard mean is:\n",meanstd.std(axis=0))
