"""
Real Python & NumPy Backend Server for Neural Network Visual Learning Lab
Serves the web dashboard and executes real Python/NumPy computations on every user action.
"""

import os
import sys
import json
import http.server
import numpy as np

PORT = int(os.environ.get("PORT", 8000))

# 2-Layer XOR Network persistent weights
xor_state = {
    "W1": np.array([[0.3, -0.4], [0.2, 0.5]], dtype=float),
    "b1": np.zeros((1, 2), dtype=float),
    "W2": np.array([[0.4], [-0.5]], dtype=float),
    "b2": np.zeros((1, 1), dtype=float),
    "epoch": 0
}

# Decision Boundary dataset & weights
boundary_data = {
    "X": np.array([
        [0.1, 0.2], [0.2, 0.9], [0.3, 0.6], [0.5, 0.8], [0.8, 0.9],
        [0.9, 0.2], [0.7, 0.1], [0.4, 0.3]
    ], dtype=float),
    "y": np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=float),
    "weights": np.array([0.0, 0.0], dtype=float),
    "bias": 0.0
}

class NeuralNetworkAPIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable caching so every request is freshly executed
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            data = json.loads(post_body.decode("utf-8"))
        except Exception:
            data = {}

        response_data = {}

        if self.path == "/api/info":
            response_data = {
                "status": "connected",
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "pid": os.getpid(),
                "engine": "CPython 3.8.8 with Native NumPy Array Operations"
            }

        elif self.path == "/api/calc_neuron":
            # Real NumPy single neuron calculation
            x = np.array([float(data.get("x1", 0)), float(data.get("x2", 0))], dtype=float)
            w = np.array([float(data.get("w1", 0)), float(data.get("w2", 0))], dtype=float)
            b = float(data.get("b", 0))
            act_type = str(data.get("activation", "step"))

            p1 = float(x[0] * w[0])
            p2 = float(x[1] * w[1])
            z = float(np.dot(x, w) + b)

            if act_type == "relu":
                out = float(np.maximum(0.0, z))
            elif act_type == "sigmoid":
                out = float(1.0 / (1.0 + np.exp(-z)))
            elif act_type == "tanh":
                out = float(np.tanh(z))
            else:  # step
                out = 1.0 if z >= 0.0 else 0.0

            response_data = {
                "p1": p1,
                "p2": p2,
                "z": z,
                "output": out,
                "executed_by": "Real NumPy (np.dot and np array calculations)"
            }

        elif self.path == "/api/train_perceptron_step":
            # Real NumPy Perceptron learning step
            x = np.array([float(data.get("x1", 0)), float(data.get("x2", 0))], dtype=float)
            target = float(data.get("target", 0))
            w = np.array([float(data.get("w1", 0)), float(data.get("w2", 0))], dtype=float)
            b = float(data.get("b", 0))
            lr = float(data.get("lr", 0.5))

            z = float(np.dot(x, w) + b)
            pred = 1.0 if z >= 0.0 else 0.0
            error = float(target - pred)

            # Weight update formulas: dw = lr * error * x
            dw = lr * error * x
            db = lr * error

            new_w = w + dw
            new_b = b + db

            response_data = {
                "z": z,
                "prediction": int(pred),
                "error": int(error),
                "dw1": float(dw[0]),
                "dw2": float(dw[1]),
                "db": float(db),
                "new_w1": float(new_w[0]),
                "new_w2": float(new_w[1]),
                "new_b": float(new_b),
                "executed_by": "Real NumPy Array Update Engine"
            }

        elif self.path == "/api/calc_loss":
            # Real NumPy Loss & Error computation
            x = float(data.get("x", 2.0))
            y = float(data.get("y", 3.0))
            w = float(data.get("w", 0.5))

            pred = float(w * x)
            diff = float(pred - y)
            loss = float(diff ** 2)

            # Generate loss curve points using NumPy linspace
            w_axis = np.linspace(-1.0, 4.0, 80)
            loss_axis = (w_axis * x - y) ** 2

            response_data = {
                "prediction": pred,
                "difference": diff,
                "loss": loss,
                "optimal_w": float(y / x) if x != 0 else 0.0,
                "curve_w": w_axis.tolist(),
                "curve_loss": loss_axis.tolist(),
                "executed_by": "NumPy np.linspace and Vectorized Power Evaluation"
            }

        elif self.path == "/api/gradient_step":
            # Real Python / NumPy gradient evaluation
            cur_x = float(data.get("x", 2.0))
            lr = float(data.get("lr", 0.2))

            # f(x) = x^2, df/dx = 2x
            gradient = float(2.0 * cur_x)
            step_size = float(lr * gradient)
            new_x = float(cur_x - step_size)

            response_data = {
                "current_x": cur_x,
                "current_y": float(cur_x ** 2),
                "gradient": gradient,
                "step_size": step_size,
                "new_x": new_x,
                "executed_by": "Python Derivative Evaluation"
            }

        elif self.path == "/api/train_boundary":
            epochs = int(data.get("epochs", 1))
            X = boundary_data["X"]
            y = boundary_data["y"]
            w = boundary_data["weights"]
            b = boundary_data["bias"]
            lr = 0.15

            for _ in range(epochs):
                for xi, yi in zip(X, y):
                    p = float(1.0 / (1.0 + np.exp(-(np.dot(xi, w) + b))))
                    err = p - yi
                    w -= lr * err * xi
                    b -= lr * err

            boundary_data["weights"] = w
            boundary_data["bias"] = b

            response_data = {
                "w1": float(w[0]),
                "w2": float(w[1]),
                "bias": float(b),
                "executed_by": "NumPy Vectorized Logistic Regression Update"
            }

        elif self.path == "/api/train_xor":
            epochs = int(data.get("epochs", 50))
            reset = bool(data.get("reset", False))

            if reset:
                xor_state["W1"] = np.random.randn(2, 2) * 0.5
                xor_state["b1"] = np.zeros((1, 2))
                xor_state["W2"] = np.random.randn(2, 1) * 0.5
                xor_state["b2"] = np.zeros((1, 1))
                xor_state["epoch"] = 0

            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
            y = np.array([0, 1, 1, 0], dtype=float)

            W1 = xor_state["W1"]
            b1 = xor_state["b1"]
            W2 = xor_state["W2"]
            b2 = xor_state["b2"]
            lr = 0.5

            total_loss = 0.0
            for _ in range(epochs):
                total_loss = 0.0
                for xi, target in zip(X, y):
                    xi = xi.reshape(1, -1)
                    # Forward
                    z1 = xi @ W1 + b1
                    a1 = 1.0 / (1.0 + np.exp(-z1))
                    z2 = a1 @ W2 + b2
                    out = 1.0 / (1.0 + np.exp(-z2))

                    err = out - target
                    total_loss += float(err[0, 0] ** 2)

                    # Backprop
                    d_out = 2 * err * out * (1.0 - out)
                    dW2 = a1.T @ d_out
                    db2 = d_out
                    d_hidden = (d_out @ W2.T) * (a1 * (1.0 - a1))
                    dW1 = xi.T @ d_hidden
                    db1 = d_hidden

                    W1 -= lr * dW1
                    b1 -= lr * db1
                    W2 -= lr * dW2
                    b2 -= lr * db2

                xor_state["epoch"] += 1

            # Predictions on XOR dataset
            preds = []
            for xi in X:
                xi = xi.reshape(1, -1)
                a1 = 1.0 / (1.0 + np.exp(-(xi @ W1 + b1)))
                out = 1.0 / (1.0 + np.exp(-(a1 @ W2 + b2)))
                preds.append(float(out[0, 0]))

            response_data = {
                "epoch": xor_state["epoch"],
                "total_loss": total_loss,
                "preds": preds,
                "executed_by": "Real NumPy Backpropagation Matrix Multiplication (@)"
            }

        else:
            self.send_response(404)
            self.end_headers()
            return

        # Send JSON response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))

if __name__ == "__main__":
    server_address = ("", PORT)
    httpd = http.server.ThreadingHTTPServer(server_address, NeuralNetworkAPIHandler)
    print(f"Neural Network Real Python Server running on port {PORT} (PID {os.getpid()})...")
    httpd.serve_forever()
