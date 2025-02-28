from sklearn.datasets import load_iris
from sklearn.svm import SVC 
data = load_iris() 
X = data.data 
y = data.target 
model = SVC(kernel='rbf') 
model.fit(X, y) 
predictions = model.predict(X)
print(predictions)
