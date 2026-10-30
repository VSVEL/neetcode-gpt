import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # # return round(your_answer, 5)
        # def sigmoid(self, x):
        #     return 1/ (1+np.exp(-x))
        # def relu(self, x):
        #     return np.max(0,x)

        z = np.round(np.dot(x,w)+b,5)
        #print(z)
        ans = np.maximum(0,z) if activation =='relu' else  1 / (1+np.exp(-z))
        print(ans)

        return round(ans,5)
