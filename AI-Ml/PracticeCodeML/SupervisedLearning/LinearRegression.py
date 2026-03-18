# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # features
y = np.array([2, 4, 5, 4, 5])                 # target

# Train model
model = LinearRegression().fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue')  # original points
plt.plot(X, y_pred, color='red') # regression line
plt.show()