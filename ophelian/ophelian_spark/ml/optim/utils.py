import numpy as np


class LBFGS:
    """
    Minimize a function using the LBFGS algorithm.
    """

    def __init__(self, x0, func, grad, m, maxiter=100, tol=1e-5, k=1.0):
        """
        Initialize the LBFGS algorithm.

        Parameters:
        - x0: Initial point for the optimization.
        - func: Function to be minimized.
        - grad: Gradient of the function.
        - m: Number of past iterations to store.
        - maxiter: Maximum number of iterations to perform (optional, default is 100).
        - tol: Tolerance value for convergence (optional, default is 1e-5).
        """

        self.x0 = x0
        self.func = func
        self.grad = grad
        self.m = m
        self.maxiter = maxiter
        self.tol = tol
        self.k = k

        self.n = len(x0)
        self.k = 0
        self.x = x0
        self.H = np.eye(self.n)  # Initial Hessian approximation
        self.s = []  # Past direction vectors
        self.y = []  # Past gradient differences

    def minimize(self, constraint_func=None, grad_constraint_func=None):
        """
        Perform the optimization.

        Parameters:
        - constraint_func: Function for the constraints (optional).
        - grad_constraint_func: Gradient of the constraints (optional).

        Returns:
        - x: Minimum point found by the algorithm.
        """

        # Iterate until convergence or maximum number of iterations is reached
        while self.k < self.maxiter:

            # Calculate gradient
            g = self.grad(self.x)

            # Add penalty term for constraints
            if constraint_func is not None:
                g += self.k * grad_constraint_func(self.x)

            # Check for convergence
            if np.linalg.norm(g) < self.tol:
                break

            # Calculate search direction
            p = -self.H.dot(g)

            # Perform line search to find step size
            alpha = self.line_search(p, constraint_func=constraint_func)

            # Update variables
            x_new = self.x + alpha * p
            self.s.append(x_new - self.x)
            self.y.append(self.grad(x_new) - g)
            self.x = x_new

            # Update Hessian approximation using BFGS formula
            if len(self.s) > self.m:
                self.s.pop(0)
                self.y.pop(0)
            self.H = self.update_H()

            self.k += 1

        return self.x

    def update_H(self):
        """
        Update Hessian approximation using the BFGS formula.

        Returns:
        - H: Updated Hessian approximation.
        """

        rho = 1.0 / np.dot(self.y[-1], self.s[-1])
        A = np.eye(self.n) - rho * np.outer(self.s[-1], self.y[-1])
        B = np.eye(self.n) - rho * np.outer(self.y[-1], self.s[-1])
        H = A.dot(self.H).dot(B) + rho * np.outer(self.s[-1], self.s[-1])

        return H

    def line_search(self, p, constraint_func=None):
        """
        Perform line search to find step size.

        Parameters:
        - p: Search direction.
        - constraint_func: Function for the constraints (optional).

        Returns:
        - alpha: Step size.
        """

        # Initialize step size
        alpha = 1.0

        # Initialize variables for line search
        x = self.x
        f_x = self.func(x)
        g_x = self.grad(x)
        g_x_p = g_x.dot(p)

        # Check constraints
        if constraint_func is not None:
            c_x = constraint_func(x)
            f_x += self.k * c_x

        # Set initial values for line search
        f_x_alpha = f_x
        c_x_alpha = c_x

        # Perform line search
        for i in range(20):
            x_alpha = x + alpha * p
            f_x_alpha = self.func(x_alpha)
            c_x_alpha = constraint_func(x_alpha) if constraint_func is not None else 0
            if f_x_alpha <= f_x + 1e-4 * alpha * g_x_p + self.k * c_x_alpha:
                break
            alpha *= 0.5

        return alpha
