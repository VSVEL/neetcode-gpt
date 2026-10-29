class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        xold = init
        if iterations == 0 :
            return xold
        
        def derivative(x):
            return 2*x
        
        for i in range(0, iterations):
            xnew = xold - learning_rate * derivative(xold)
            xold = xnew

        return round(xold,5) 