import numpy as np
from pyspark.ml import Transformer


class LogitRegression(Transformer):
    """
    Logistic Regression classifier.

    This class implements logistic regression using NumPy for calculations.

    :param learning_rate: float, optional (default=0.01)
        The learning rate for gradient descent.
    :param num_iterations: int, optional (default=1000)
        The number of iterations for gradient descent.

    Example:
        lr = LogitRegression(learning_rate=0.01, num_iterations=1000)
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
        Fit the logistic regression model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

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
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    @staticmethod
    def _sigmoid(x):
        """
        Compute the sigmoid of x.

        :param x: numpy.ndarray
            The input array.
        :return: numpy.ndarray
            The sigmoid of the input array.
        """
        return 1 / (1 + np.exp(-x))


class RandomForestModel(Transformer):
    """
    Random Forest classifier.

    This class implements a Random Forest classifier using NumPy for calculations.

    :param num_trees: int, optional (default=100)
        The number of trees in the forest.
    :param max_depth: int, optional (default=10)
        The maximum depth of the trees.

    Example:
        rf = RandomForestModel(num_trees=100, max_depth=10)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
    """

    def __init__(self, num_trees=100, max_depth=10):
        super().__init__()
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        for _ in range(self.num_trees):
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target labels for the given input features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        tree_preds = np.array([self._predict_tree(tree, X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [
            np.bincount(tree_pred.astype(int)).argmax() for tree_pred in tree_preds
        ]
        return np.array(y_pred)

    def _build_tree(self, X, y, depth):
        """
        Build a decision tree.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        :param depth: int
            The current depth of the tree.
        :return: dict
            The decision tree.
        """
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples <= 1:
            return self._leaf(y)

        feature_idx, threshold = self._best_split(X, y, num_features)
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return {
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left,
            "right": right,
        }

    def _leaf(self, y):
        """
        Create a leaf node.

        :param y: numpy.ndarray
            The target labels.
        :return: int
            The most common label.
        """
        return np.bincount(y).argmax()

    def _best_split(self, X, y, num_features):
        """
        Find the best split for a node.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        :param num_features: int
            The number of features.
        :return: tuple
            The best feature index and threshold.
        """
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_threshold = feature_idx, threshold
        return split_idx, split_threshold

    def _information_gain(self, X, y, feature_idx, threshold):
        """
        Compute the information gain of a split.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        :param feature_idx: int
            The feature index.
        :param threshold: float
            The threshold.
        :return: float
            The information gain.
        """
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        n, n_left, n_right = len(y), len(y[left_indices]), len(y[right_indices])
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(
            y[right_indices]
        )
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        return parent_entropy - child_entropy

    @staticmethod
    def _entropy(y):
        """
        Compute the entropy of a distribution.

        :param y: numpy.ndarray
            The target labels.
        :return: float
            The entropy.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _predict_tree(self, tree, X):
        """
        Predict the labels for a given tree and input data.

        :param tree: dict
            The decision tree.
        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        if isinstance(tree, dict):
            feature_idx = tree["feature_idx"]
            threshold = tree["threshold"]
            left = tree["left"]
            right = tree["right"]

            return np.array(
                [
                    self._predict_tree(
                        left if x[feature_idx] <= threshold else right, x
                    )
                    for x in X
                ]
            )
        return np.array([tree] * X.shape[0])


class SVM(Transformer):
    """
    Support Vector Machine (SVM) classifier.

    This class implements a linear SVM classifier using NumPy for calculations.

    :param learning_rate: float, optional (default=0.001)
        The learning rate for gradient descent.
    :param lambda_param: float, optional (default=0.01)
        The regularization parameter.
    :param num_iterations: int, optional (default=1000)
        The number of iterations for gradient descent.

    Example:
        svm = SVM(learning_rate=0.001, lambda_param=0.01, num_iterations=1000)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the SVM model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y[idx])
                    )
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X):
        """
        Predict the target labels for the given input features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)


class KNN(Transformer):
    """
    K-Nearest Neighbors (KNN) classifier.

    This class implements a KNN classifier using NumPy for calculations.

    :param k: int, optional (default=3)
        The number of neighbors to use.

    Example:
        knn = KNN(k=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    """

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the KNN model.

        :param X: numpy.ndarray
            The input features.
        :param y: numpy.ndarray
            The target labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the target labels for the given input features.

        :param X: numpy.ndarray
            The input features.
        :return: numpy.ndarray
            The predicted labels.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predict the label for a single input.

        :param x: numpy.ndarray
            A single input feature.
        :return: int
            The predicted label.
        """
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
