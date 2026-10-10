"""
Step 1: Single Neuron Basics

This file explains the smallest building block of a neural network:
A neuron with input(s), weights, bias, activation, and output.

A neuron computes:
    z = x1*w1 + x2*w2 + ... + b
    a = activation(z)

We will use a simple threshold activation to learn a basic rule.
"""

import numpy as np

# Example dataset for AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([0, 0, 0, 1], dtype=float)

# Weights and bias for one neuron
weights = np.array([0.5, 0.5])
bias = -0.7


def step_activation(z):
    """Threshold activation: 1 if z >= 0, else 0."""
    return 1 if z >= 0 else 0


print("=== Single Neuron Example ===")
print("We compute: output = step(x1*w1 + x2*w2 + b)")
print("Current weights:", weights)
print("Current bias:", bias)
print()

for i, (x, target) in enumerate(zip(X, y), start=1):
    z = np.dot(x, weights) + bias
    prediction = step_activation(z)
    print(f"Sample {i}: x={x}, target={int(target)}, z={z:.2f}, prediction={prediction}")

print("\nInterpretation:")
print("- The neuron multiplies each input by a weight.")
print("- It adds a bias.")
print("- It applies an activation function.")
print("- For the AND gate, the neuron should output 1 only when both inputs are 1.")
print("- In real training, the weights and bias adjust automatically.")
