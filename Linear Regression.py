# Linear Regression using Stochastic Gradient Descent (SGD)
# From scratch — no ML libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset
# Assuming dataset has two columns: X (feature) and Y (target)
data = pd.read_csv('data.csv')
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Step 2: Normalize data (optional but helps convergence)
X = (X - np.mean(X)) / np.std(X)

# Step 3: Initialize parameters
m = 0.0  # slope
b = 0.0  # intercept
learning_rate = 0.01
epochs = 100  # total iterations over data

# Step 4: Stochastic Gradient Descent loop
n = len(X)

for epoch in range(epochs):
    for i in range(n):
        xi = X[i]
        yi = y[i]
        y_pred = m * xi + b
        error = yi - y_pred

        # Gradient update for one sample
        m += learning_rate * error * xi
        b += learning_rate * error

    # Optional: print loss occasionally
    if (epoch + 1) % 10 == 0:
        y_pred_all = m * X + b
        loss = np.mean((y - y_pred_all) ** 2)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Step 5: Results
print(f"\nFinal model: y = {m:.4f}x + {b:.4f}")

# Step 6: Visualization
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + b, color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression using SGD (from scratch)')
plt.legend()
plt.show()
