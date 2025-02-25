from sklearn.linear_model import LinearRegression 
import numpy as np 
# Example dataset 
X = np.array([[1], [2], [3], [4], [5]]) # Independent variable 
y = np.array([2, 4, 5, 4, 5]) # Dependent variable 
# Initialize and fit the model 
model = LinearRegression() 
model.fit(X, y) 
# Coefficients 
print("Slope (m):", model.coef_[0]) 
print("Intercept (c):", model.intercept_) 
# Make predictions 
predictions = model.predict(X) 
print("Predictions:", predictions)
