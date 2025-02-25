from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris 
data = load_iris() 
X, y = data.data, data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred)) 
