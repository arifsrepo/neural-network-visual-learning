"""
Step 3: Understanding Loss

Loss tells us how wrong the prediction is.
For a single weight, we can visualize how the loss changes.
"""

import numpy as np
import matplotlib.pyplot as plt

# Suppose the true target is 3.0.
# We predict y = w * x, with x = 2.
true_target = 3.0
x = 2.0


# MSE loss: (pred - target)^2

def mse_loss(weight):
    prediction = weight * x
    return (prediction - true_target) ** 2

weights = np.linspace(-2, 6, 200)
losses = [mse_loss(w) for w in weights]

print("=== Loss Function Example ===")
print("If the true target is", true_target, "and x =", x)
print("We use prediction = w * x")
print("Loss = (prediction - true_target)^2")
print("Best weight is near 1.5 (since 1.5 * 2 = 3)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(weights, losses, label='MSE Loss')
ax.axvline(1.5, color='red', linestyle='--', label='Best weight = 1.5')
ax.set_title('Loss vs Weight')
ax.set_xlabel('Weight w')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)
plt.show()

print("\nKey idea:")
print("The model learns by adjusting weights to reduce loss.")
print("The best weight is the one with the minimum loss.")
