import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        activation = x
        num_layers = len(weights)

        for i, (w, b) in enumerate(zip(weights, biases)):
            # Handle orientation: if weights are shaped (out_features, in_features),
            # np.dot(w, activation) is used. If shaped (in_features, out_features),
            # np.dot(activation, w) is used.
            if activation.shape[0] == w.shape[0]:
                z = np.dot(activation, w) + b
            else:
                z = np.dot(w, activation) + b

            # Apply ReLU for all layers except the last one
            if i < num_layers - 1:
                activation = np.maximum(0.0, z)
            else:
                activation = z

        return np.round(activation, 5)
