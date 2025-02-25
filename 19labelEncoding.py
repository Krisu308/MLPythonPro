import pandas as pd
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("D:\\2189\\BCAresult.csv")
print(data.head())
lblenc=LabelEncoder()
data["name_new"]=lblenc.fit_transform(data["name"])
