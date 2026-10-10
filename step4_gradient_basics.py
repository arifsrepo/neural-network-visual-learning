"""
Step 4: Gradient Basics

A gradient tells us which direction to move to reduce loss.
We will use a very simple function: y = x^2

At x = 2, the derivative is 4.
That means moving x to the left (decreasing x) reduces the value.
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 2


x = np.linspace(-3, 3, 200)
y = f(x)

current_x = 2.0
current_y = f(current_x)
gradient = 2 * current_x

print("=== Gradient Example ===")
print("Function: y = x^2")
print("Current x =", current_x)
print("Current y =", current_y)
print("Derivative (gradient) =", gradient)
print("The gradient tells us the slope at this point.")
print("If the gradient is positive, decreasing x will reduce the value.")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y, label='y = x^2')
ax.plot(current_x, current_y, 'ro', label='Current point')
ax.axvline(current_x, color='gray', linestyle='--')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Gradient of x^2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)
plt.show()

print("\nIn neural networks, the model updates weights using gradients.")
print("It moves weights in the direction that decreases loss.")
