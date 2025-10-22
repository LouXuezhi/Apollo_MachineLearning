import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
# Generate some example data

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5   
# Create a Linear Regression model
model = LinearRegression()
# Fit the model to the data
model.fit(X, y)
# Make predictions
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)


# Plot the results
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_new, y_predict, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
# Display the coefficients
print(f'Intercept: {model.intercept_[0]}')
print(f'Coefficient: {model.coef_[0][0]}')      
# Create a DataFrame to show the results
results = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})
print(results.head())
    