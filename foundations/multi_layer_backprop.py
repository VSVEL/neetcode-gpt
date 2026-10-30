import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        # Convert inputs to NumPy arrays
        x_arr = np.array(x, dtype=np.float64)
        W1_arr = np.array(W1, dtype=np.float64)
        b1_arr = np.array(b1, dtype=np.float64)
        W2_arr = np.array(W2, dtype=np.float64)
        b2_arr = np.array(b2, dtype=np.float64)
        y_true_arr = np.array(y_true, dtype=np.float64)

        # ----------------- Forward Pass -----------------
        # Layer 1: z1 = W1 @ x + b1
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0, z1)

        # Layer 2: z2 = W2 @ a1 + b2
        z2 = np.dot(W2_arr, a1) + b2_arr
        y_pred = z2

        # Loss: Mean Squared Error
        loss = float(np.mean((y_pred - y_true_arr) ** 2))

        # ----------------- Backward Pass -----------------
        K = len(y_true_arr)

        # Output layer gradient
        delta2 = (2.0 / K) * (y_pred - y_true_arr)

        # Layer 2 gradients (shape: (output_dim, hidden_dim))
        dW2 = np.outer(delta2, a1)
        db2 = delta2

        # Backpropagation into Layer 1
        da1 = np.dot(W2_arr.T, delta2)
        delta1 = da1 * (z1 > 0)

        # Layer 1 gradients (shape: (hidden_dim, input_dim))
        dW1 = np.outer(delta1, x_arr)
        db1 = delta1

        return {
            'loss': round(loss, 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
        

