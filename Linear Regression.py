
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

X = (X - np.mean(X)) / np.std(X)

m = 0.0 
b = 0.0  
learning_rate = 0.01
epochs = 100 

n = len(X)

for epoch in range(epochs):
    for i in range(n):
        xi = X[i]
        yi = y[i]
        y_pred = m * xi + b
        error = yi - y_pred

       
        m += learning_rate * error * xi
        b += learning_rate * error

    
    if (epoch + 1) % 10 == 0:
        y_pred_all = m * X + b
        loss = np.mean((y - y_pred_all) ** 2)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


print(f"\nFinal model: y = {m:.4f}x + {b:.4f}")


plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + b, color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression using SGD (from scratch)')
plt.legend()
plt.show()

