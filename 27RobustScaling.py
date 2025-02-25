from pandas import *
from sklearn.preprocessing import RobustScaler
import seaborn as sb
import matplotlib.pyplot as plt;
#d=read_csv("D:\\2189\\BCAresult.csv")
#print(d.head())
#data=get_dummies(d["name"])
#print(data.head())
data=([[22,33,44],[23,6,7],[88,7,78]])
print(data)
scaler=RobustScaler()
robust_scaled_data=scaler.fit_transform(data)
print(robust_scaled_data)
"""for i in d.select_dtypes(include="number").columns:
    sb.histplot(data=d,x=i)
    plt.show()"""
