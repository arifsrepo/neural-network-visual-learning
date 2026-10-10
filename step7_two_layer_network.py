"""
Step 7: Two-Layer Network

This is the first time we build a tiny neural network with a hidden layer.
It is still simple, but now the network can represent more complex patterns.

We use a tiny XOR-style dataset.
"""

import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([0, 1, 1, 0], dtype=float)

# One hidden layer with 2 neurons.
# Input -> Hidden -> Output
np.random.seed(0)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(x):
    z1 = x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    output = sigmoid(z2)
    return z1, a1, z2, output


print("=== Two-Layer Network ===")
print("This network has 2 hidden neurons and 1 output neuron")
print("The goal is to learn XOR.")
print()

for epoch in range(3000):
    total_loss = 0.0
    for x, target in zip(X, y):
        x = x.reshape(1, -1)
        z1, a1, z2, output = forward(x)

        # loss = (output - target)^2
        error = output - target
        total_loss += error ** 2

        # Backpropagation (simplified for learning)
        d_output = 2 * error * output * (1 - output)
        dW2 = a1.T @ d_output
        db2 = d_output
        d_hidden = (d_output @ W2.T) * (a1 * (1 - a1))
        dW1 = x.T @ d_hidden
        db1 = d_hidden

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={total_loss:.4f}")

print("\nFinal predictions:")
for x, target in zip(X, y):
    _, _, _, output = forward(x.reshape(1, -1))
    print(f"Input={x}, target={target}, prediction={output[0, 0]:.3f}")

print("\nKey idea:")
print("A hidden layer allows the network to learn patterns that a single neuron cannot.")
print("This is the start of deeper neural networks.")
