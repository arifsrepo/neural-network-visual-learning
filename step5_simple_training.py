"""
Step 5: Simple Training Loop

This is the first real learning example.
We will train a single neuron to learn the AND gate.

The neuron uses:
    output = 1 if x1*w1 + x2*w2 + b >= 0 else 0

We update weights when the prediction is wrong.
"""

import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([0, 0, 0, 1], dtype=float)

weights = np.array([0.0, 0.0], dtype=float)
bias = 0.0
learning_rate = 0.5


def predict(x, w, b):
    z = np.dot(x, w) + b
    return 1 if z >= 0 else 0

print("=== Perceptron Training ===")
print("Training a single neuron to learn the AND gate")
print("Initial weights:", weights)
print("Initial bias:", bias)
print()

for epoch in range(20):
    errors = 0
    for x, target in zip(X, y):
        prediction = predict(x, weights, bias)
        if prediction != target:
            weights += learning_rate * (target - prediction) * x
            bias += learning_rate * (target - prediction)
            errors += 1

    print(f"Epoch {epoch + 1}: weights={weights}, bias={bias}, errors={errors}")

    if errors == 0:
        print("Model has learned the AND gate.")
        break

print("\nFinal predictions:")
for x in X:
    print(x, "->", predict(x, weights, bias))

print("\nThis is the basic idea behind neural network learning:")
print("- Make a prediction")
print("- Compare to the target")
print("- Adjust weights if wrong")
print("- Repeat many times")
