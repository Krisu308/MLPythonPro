from pandas import *
from sklearn.preprocessing import StandardScaler
import seaborn as sb
import matplotlib.pyplot as plt;
d=read_csv("D:\\2189\\BCAresult.csv")
print(d.head())
data=get_dummies(d["name"])
print(data.head())
scaler = StandardScaler() 
standardized_data = scaler.fit_transform(data) 
print(standardized_data) 
for i in d.select_dtypes(include="number").columns:
    sb.histplot(data=d,x=i)
    plt.show()
