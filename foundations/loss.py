import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Compute the loss using your formula
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return round(loss,4)

        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # 2. Compute the element-wise product: y_ic * ln(p_ic)
        elementwise_loss = y_true * np.log(y_pred)

        # 3. Sum over classes (c=1 to C), then average over samples (n)
        # np.sum(..., axis=1) handles the inner sum over classes
        # np.mean(...) handles the outer 1/n sum over samples
        loss = -np.mean(np.sum(elementwise_loss, axis=1))

        return round(loss,4)
