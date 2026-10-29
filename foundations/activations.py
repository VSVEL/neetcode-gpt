import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # NumPy applies operations element-wise natively
        res = 1 / (1 + np.exp(-z))
    
        return np.round(res, 5)


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        res = []
        for element in np.nditer(z):
            x = max(0.0,element)
            res.append(x)
        return res

        
