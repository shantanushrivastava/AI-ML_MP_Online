# Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Ice cream sales data
temp = np.array([20, 22, 25, 28, 30, 32, 35, 37, 40]).reshape(-1, 1)
sales = np.array([30, 35, 60, 85, 110, 130, 140, 145, 147])

# Convert temp to polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
temp_poly = poly.fit_transform(temp)

# Train the model
model = LinearRegression()
model.fit(temp_poly, sales)

# Make predictions
predicted = model.predict(temp_poly)


# Plot
plt.scatter(temp, sales, color='red', label='Actual sales')
plt.plot(temp, predicted, color='blue', label='Predicted sales')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sold')
plt.title('Ice Cream Sales vs Temperature')
plt.legend()
plt.show()