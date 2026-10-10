"""
Step 6: Decision Boundary Animation

We train a simple linear classifier on 2D data.
The decision boundary is the line that separates classes.
As training continues, the line moves.
"""

import numpy as np
import matplotlib.pyplot as plt

# Two classes in 2D space
X = np.array([
    [0.1, 0.2], [0.2, 0.9], [0.3, 0.6], [0.5, 0.8], [0.8, 0.9],
    [0.9, 0.2], [0.7, 0.1], [0.4, 0.3]
], dtype=float)

y = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=float)

weights = np.array([0.0, 0.0], dtype=float)
bias = 0.0
learning_rate = 0.1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, w, b):
    z = float(np.dot(x, w) + b)
    return sigmoid(z)

fig, ax = plt.subplots(figsize=(7, 6))
for epoch in range(200):
    for x, target in zip(X, y):
        p = predict(x, weights, bias)
        error = p - target
        weights -= learning_rate * error * x
        bias -= learning_rate * error

    # Plot boundary after each 20 epochs
    if epoch % 20 == 0:
        xs = np.linspace(0, 1, 50)
        ys = [-(weights[0] * x + bias) / weights[1] if weights[1] != 0 else 0 for x in xs]
        ax.clear()
        ax.scatter(X[:5, 0], X[:5, 1], c='blue', label='Class 0')
        ax.scatter(X[5:, 0], X[5:, 1], c='red', label='Class 1')
        ax.plot(xs, ys, color='black', linewidth=2, label='Decision Boundary')
        ax.set_title(f"Decision Boundary after {epoch + 1} epochs")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        plt.pause(0.1)

print("Final weights:", weights)
print("Final bias:", bias)
plt.show()

print("\nA decision boundary is the line that separates two classes.")
print("In a neural network, the boundary moves as weights change during training.")
