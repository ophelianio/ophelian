import numpy as np
from pyspark.ml import Transformer


class LinearRegression(Transformer):
    """
    Linear Regression using gradient descent.

    :param learning_rate: float, optional (default=0.01)
        The learning rate for gradient descent.
    :param num_iterations: int, optional (default=1000)
        The number of iterations for gradient descent.

    Example:
        lr = LinearRegression(learning_rate=0.01, num_iterations=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Linear Regression model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the target labels for the given input features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        return np.dot(X, self.weights) + self.bias


class RidgeRegression(Transformer):
    """
    Ridge Regression using gradient descent.

    :param learning_rate: float, optional (default=0.01)
        The learning rate for gradient descent.
    :param num_iterations: int, optional (default=1000)
        The number of iterations for gradient descent.
    :param lambda_param: float, optional (default=0.01)
        The regularization parameter.

    Example:
        rr = RidgeRegression(learning_rate=0.01, num_iterations=1000, lambda_param=0.01)
        rr.fit(X_train, y_train)
        y_pred = rr.predict(X_test)
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Ridge Regression model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y)) + (
                self.lambda_param * self.weights
            )
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the target labels for the given input features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        return np.dot(X, self.weights) + self.bias


class LassoRegression(Transformer):
    """
    Lasso Regression using gradient descent.

    :param learning_rate: float, optional (default=0.01)
        The learning rate for gradient descent.
    :param num_iterations: int, optional (default=1000)
        The number of iterations for gradient descent.
    :param lambda_param: float, optional (default=0.01)
        The regularization parameter.

    Example:
        lr = LassoRegression(learning_rate=0.01, num_iterations=1000, lambda_param=0.01)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Lasso Regression model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(
                X.T, (y_predicted - y)
            ) + self.lambda_param * np.sign(self.weights)
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the target labels for the given input features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        return np.dot(X, self.weights) + self.bias


class PolynomialRegression(Transformer):
    """
    Polynomial Regression using gradient descent.

    :param degree: int, optional (default=2)
        The degree of the polynomial features.
    :param learning_rate: float, optional (default=0.01)
        The learning rate for gradient descent.
    :param num_iterations: int, optional (default=1000)
        The number of iterations for gradient descent.

    Example:
        pr = PolynomialRegression(degree=2, learning_rate=0.01, num_iterations=1000)
        pr.fit(X_train, y_train)
        y_pred = pr.predict(X_test)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        super().__init__()
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _polynomial_features(self, X):
        """
        Generate polynomial features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The polynomial features.
        """
        from itertools import combinations_with_replacement

        n_samples, n_features = X.shape
        combinations = list(
            combinations_with_replacement(range(n_features), self.degree)
        )
        n_output_features = len(combinations)
        X_poly = np.empty((n_samples, n_output_features))

        for i, comb in enumerate(combinations):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)

        return X_poly

    def fit(self, X, y):
        """
        Fit the Polynomial Regression model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        X_poly = self._polynomial_features(X)
        num_samples, num_features = X_poly.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(X_poly, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X_poly.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the target labels for the given input features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias
