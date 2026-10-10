"""
Step 2: Visualize Forward Pass

A single neuron receives inputs and produces one output.
We will print the calculation clearly and also show a simple visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

# Example input sample
x = np.array([1.0, 0.5])
weights = np.array([2.0, 1.5])
bias = -0.5


def relu(z):
    return max(0.0, z)

z = np.dot(x, weights) + bias
output = relu(z)

print("=== Forward Pass Visualization ===")
print("Inputs:", x)
print("Weights:", weights)
print("Bias:", bias)
print("Weighted sum z = x·w + b =", z)
print("Activation output = ReLU(z) =", output)

# Simple visual explanation
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title("Single Neuron Forward Pass")
ax.text(0.05, 0.8, "Inputs", fontsize=12)
ax.text(0.35, 0.8, "Weights", fontsize=12)
ax.text(0.62, 0.8, "Weighted Sum", fontsize=12)
ax.text(0.82, 0.8, "Output", fontsize=12)

ax.plot([0.1, 0.25], [0.7, 0.7], color='black')
ax.plot([0.3, 0.45], [0.7, 0.7], color='black')
ax.plot([0.6, 0.75], [0.7, 0.7], color='black')
ax.plot([0.8, 0.92], [0.7, 0.7], color='black')

ax.text(0.08, 0.7, f"{x[0]:.1f}", fontsize=12)
ax.text(0.08, 0.45, f"{x[1]:.1f}", fontsize=12)
ax.text(0.34, 0.7, f"{weights[0]:.1f}", fontsize=12)
ax.text(0.34, 0.45, f"{weights[1]:.1f}", fontsize=12)
ax.text(0.60, 0.7, f"{z:.2f}", fontsize=12)
ax.text(0.82, 0.7, f"{output:.2f}", fontsize=12)

ax.text(0.1, 0.15, "z = x1*w1 + x2*w2 + b", fontsize=12)
ax.text(0.1, 0.05, "a = ReLU(z)", fontsize=12)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.show()
