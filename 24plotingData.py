import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
OriginalData=pd.read_csv("D:\\2189\\BCAresult.csv")
#OriginalData=pd.read_csv("https://raw.githubusercontent.com/Krisu308/Machine-Learning-python/refs/heads/main/bank_transactions_data_2.csv")
print(OriginalData.head())
for i in OriginalData.select_dtypes(include="number").columns:
    sb.histplot(data=OriginalData,x=i)
plt.show()
    
