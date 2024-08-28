import numpy as np
from sklearn import linear_model

# Define the data
height = np.array([[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
weight = np.array([8, 10, 12, 14, 16, 18, 20])

# Create a linear regression model
reg = linear_model.LinearRegression()

# Fit the model
reg.fit(height, weight)

# Predict weight for a given height
X_height = np.array([[12.0]])
predicted_weight = reg.predict(X_height)

# Print the predicted weight
print(predicted_weight)
