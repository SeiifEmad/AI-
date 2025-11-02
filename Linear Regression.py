import numpy as np
import matplotlib.pyplot as plt

x_data, y_data = np.loadtxt('data.txt', delimiter=',', unpack=True)


m = 0.0
b = 0.0
learning_rate = 0.001
epochs = 50  
n = len(x_data)


for epoch in range(epochs):
    for i in range(n):
        x_i = x_data[i]
        y_i = y_data[i]

       
        y_pred = m * x_i + b
        error = y_pred - y_i

      
        dm = 2 * error * x_i
        db = 2 * error

      
        m -= learning_rate * dm
        b -= learning_rate * db


y_pred = m * x_data + b


print(f"Final slope (m): {m:.4f}   salary increase per year (k$)")
print(f"Final intercept (b): {b:.4f}   base salary (k$)")


years = 21
predicted_salary = m * years + b
print(f"\nPredicted salary for {years} years of experience: ${predicted_salary:.2f}k/month")


plt.scatter(x_data, y_data, color='blue', label='Actual Data')
plt.plot(x_data, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Monthly Salary ( $)')
plt.title('Linear Regression : Experience vs Salary')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
