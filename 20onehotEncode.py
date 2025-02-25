import pandas as pd
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("D:\\2189\\BCAresult.csv")
print(data.head())
lblenc=LabelEncoder()
newdata=pd.get_dummies(data["name"])
newdata
