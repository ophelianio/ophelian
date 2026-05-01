import numpy as np
from pyspark.ml import Transformer
from pyspark.sql import Row


class KMeansCluster(Transformer):
    """
    K-Means clustering using NumPy.

    K-Means clustering is an unsupervised learning algorithm used to partition a dataset into K distinct,
    non-overlapping subsets or clusters.

    The algorithm iteratively assigns each data point to the nearest cluster center (centroid),
    then updates the centroids based on the mean of the points in each cluster.

    This process repeats until the centroids stabilize (convergence) or a maximum number of iterations is reached.

    :param k: Number of clusters.
    :param max_iter: Maximum number of iterations. Default is 300.
    :param tol: Tolerance for convergence. Default is 1e-4.

    Example:

        from your_module import KMeansCluster

        kmeans = KMeansCluster(k=3)
        model = kmeans.fit(data)
        clustered_data = model._transform(data)
    """

    def __init__(self, k, max_iter=300, tol=1e-4):
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array(
                [X[labels == j].mean(axis=0) for j in range(self.k)]
            )

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids

        return self

    def _transform(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        rows = [Row(cluster=int(label)) for label in labels]
        return data.sql_ctx.createDataFrame(rows)


class FuzzyClusterMeans(Transformer):
    """
    Fuzzy C-Means clustering using NumPy.

    Fuzzy C-Means clustering allows each data point to belong to multiple
    clusters with varying degrees of membership.
    The algorithm minimizes the objective function by iteratively updating
    the membership matrix and the cluster centroids.
    The fuzziness parameter (m) controls the degree of fuzziness.

    :param k: Number of clusters.
    :param m: Fuzziness parameter.
    :param max_iter: Maximum number of iterations. Default is 300.
    :param tol: Tolerance for convergence. Default is 1e-4.

    Example:

        from your_module import FuzzyClusterMeans

        fuzzy_cmeans = FuzzyClusterMeans(k=3, m=2)
        model = fuzzy_cmeans.fit(data)
        clustered_data = model._transform(data)
    """

    def __init__(self, k, m, max_iter=300, tol=1e-4):
        super().__init__()
        self.k = k
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.membership = None

    def fit(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        n_samples, n_features = X.shape
        self.membership = np.random.dirichlet(np.ones(self.k), size=n_samples)
        for _ in range(self.max_iter):
            self.centroids = (self.membership.T**self.m @ X) / np.sum(
                self.membership.T**self.m, axis=1, keepdims=True
            ).T
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            new_membership = 1.0 / np.sum(
                (distances[:, :, np.newaxis] / distances[:, np.newaxis, :])
                ** (2 / (self.m - 1)),
                axis=2,
            )
            if np.linalg.norm(new_membership - self.membership) < self.tol:
                break
            self.membership = new_membership
        return self

    def _transform(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        rows = [Row(cluster=int(label)) for label in labels]
        return data.sql_ctx.createDataFrame(rows)


class BKM(Transformer):
    """
    Bisecting K-Means clustering using NumPy.

    Bisecting K-Means is a hierarchical clustering method that recursively
    applies K-Means clustering to split clusters into two until the desired number of clusters is reached.

    :param k: Number of clusters. Default is 2.
    :param max_iter: Maximum number of iterations. Default is 20.
    :param seed: Random seed. Default is None.
    :param features_col: Name of the features column. Default is 'features'.

    Example:

        from your_module import BKM

        bkm = BKM(k=5)
        model = bkm.fit(data)
        clustered_data = bkm._transform(data)
    """

    def __init__(self, k=2, max_iter=20, seed=None, features_col="features"):
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.features_col = features_col
        self.model = None

    def fit(self, data):
        X = np.array(
            data.select(self.features_col)
            .rdd.map(lambda row: row[0].toArray())
            .collect()
        )
        clusters = [(X, np.arange(len(X)))]
        while len(clusters) < self.k:
            max_cluster, max_indices = max(clusters, key=lambda c: len(c[1]))
            kmeans = KMeansCluster(2, self.max_iter).fit(max_cluster)
            distances = np.linalg.norm(
                max_cluster[:, np.newaxis] - kmeans.centroids, axis=2
            )
            labels = np.argmin(distances, axis=1)
            clusters.remove((max_cluster, max_indices))
            for j in range(2):
                clusters.append((max_cluster[labels == j], max_indices[labels == j]))
        self.model = clusters
        return self

    def _transform(self, data):
        X = np.array(
            data.select(self.features_col)
            .rdd.map(lambda row: row[0].toArray())
            .collect()
        )
        labels = np.zeros(len(X), dtype=int)
        for i, (cluster, indices) in enumerate(self.model):
            labels[indices] = i
        rows = [Row(cluster=int(label)) for label in labels]
        return data.sql_ctx.createDataFrame(rows)

    def get_cluster_centers(self):
        return [c[0].mean(axis=0) for c, _ in self.model]

    def get_cluster_sizes(self):
        return [len(c[1]) for _, c in self.model]


class GaussianM(Transformer):
    """
    Gaussian Mixture Model clustering using NumPy.

    Gaussian Mixture Models (GMMs) are a probabilistic model that assumes
    that the data is generated from a mixture of several Gaussian distributions with unknown parameters.

    :param n_components: Number of mixture components.
    :param max_iter: Maximum number of iterations. Default is 100.
    :param tol: Tolerance for convergence. Default is 1e-4.

    Example:

        from your_module import GaussianM

        gmm = GaussianM(n_components=3)
        model = gmm.fit(data)
        clustered_data = model._transform(data)
    """

    def __init__(self, n_components, max_iter=100, tol=1e-4):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None

    def fit(self, data):
        """
        Fit the Gaussian Mixture Model to the data.

        :param data: Input data to cluster.
        :return: Fitted Gaussian Mixture Model.

        Example:

            model = gmm.fit(data)
        """
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)
        log_likelihood = 0
        for _ in range(self.max_iter):
            likelihood = np.array(
                [
                    self.weights[k]
                    * self.multivariate_gaussian(X, self.means[k], self.covariances[k])
                    for k in range(self.n_components)
                ]
            )
            log_likelihood_new = np.sum(np.log(np.sum(likelihood, axis=0)))
            likelihood /= np.sum(likelihood, axis=0)
            self.weights = likelihood.mean(axis=1)
            self.means = (
                np.dot(likelihood, X) / np.sum(likelihood, axis=1, keepdims=True).T
            )
            self.covariances = np.array(
                [
                    np.dot(
                        (likelihood[k][:, np.newaxis] * (X - self.means[k])).T,
                        X - self.means[k],
                    )
                    / likelihood[k].sum()
                    for k in range(self.n_components)
                ]
            )
            if abs(log_likelihood_new - log_likelihood) < self.tol:
                break
            log_likelihood = log_likelihood_new
        return self

    @staticmethod
    def multivariate_gaussian(X, mean, covariance):
        """
        Compute the multivariate Gaussian distribution.

        :param X: Input data.
        :param mean: Mean vector.
        :param covariance: Covariance matrix.
        :return: Multivariate Gaussian distribution.

        Example:

            probabilities = gmm.multivariate_gaussian(X, mean, covariance)
        """
        n = X.shape[1]
        diff = X - mean
        return np.exp(
            -0.5 * np.sum(np.dot(diff, np.linalg.inv(covariance)) * diff, axis=1)
        ) / np.sqrt(((2 * np.pi) ** n) * np.linalg.det(covariance))

    def _transform(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        likelihood = np.array(
            [
                self.weights[k]
                * self.multivariate_gaussian(X, self.means[k], self.covariances[k])
                for k in range(self.n_components)
            ]
        )
        labels = np.argmax(likelihood, axis=0)
        rows = [Row(cluster=int(label)) for label in labels]
        return data.sql_ctx.createDataFrame(rows)


class LatentDirichlet(Transformer):
    """
    Latent Dirichlet Allocation (LDA) using NumPy.

    LDA is a generative statistical model that explains a set of
    observations through unobserved groups, which explain why some parts of the data are similar.

    :param n_topics: Number of topics.
    :param max_iter: Maximum number of iterations. Default is 100.
    :param tol: Tolerance for convergence. Default is 1e-4.

    Example:

        from your_module import LatentDirichlet

        lda = LatentDirichlet(n_topics=10)
        model = lda.fit(data)
        topic_distribution = model._transform(data)
    """

    def __init__(self, n_topics, max_iter=100, tol=1e-4):
        super().__init__()
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.topic_word_distribution = None
        self.document_topic_distribution = None

    def fit(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        n_samples, n_features = X.shape
        self.topic_word_distribution = np.random.dirichlet(
            np.ones(n_features), self.n_topics
        )
        self.document_topic_distribution = np.random.dirichlet(
            np.ones(self.n_topics), n_samples
        )
        for _ in range(self.max_iter):
            topic_distribution = np.dot(
                self.document_topic_distribution[:, :, np.newaxis],
                self.topic_word_distribution[np.newaxis, :, :],
            )
            topic_distribution /= topic_distribution.sum(axis=1, keepdims=True)
            self.document_topic_distribution = topic_distribution.sum(axis=2)
            self.document_topic_distribution /= self.document_topic_distribution.sum(
                axis=1, keepdims=True
            )
            self.topic_word_distribution = np.dot(topic_distribution.sum(axis=0).T, X)
            self.topic_word_distribution /= self.topic_word_distribution.sum(
                axis=1, keepdims=True
            )
        return self

    def _transform(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        topic_distribution = np.dot(
            self.document_topic_distribution[:, :, np.newaxis],
            self.topic_word_distribution[np.newaxis, :, :],
        )
        topic_distribution /= topic_distribution.sum(axis=1, keepdims=True)
        labels = np.argmax(topic_distribution.sum(axis=2), axis=1)
        rows = [Row(topic=int(label)) for label in labels]
        return data.sql_ctx.createDataFrame(rows)


class ClusterFeatureTree(Transformer):
    """
    BIRCH clustering using NumPy.

    BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
    is an efficient clustering algorithm for large datasets.

    :param threshold: Threshold for the BIRCH algorithm.
    :param branching_factor: Branching factor for the BIRCH algorithm.
    :param n_clusters: Number of clusters.

    Example:

        from your_module import ClusterFeatureTree

        birch = ClusterFeatureTree(threshold=0.5, branching_factor=50, n_clusters=3)
        model = birch.fit(data)
        clustered_data = model._transform(data)
    """

    def __init__(self, threshold, branching_factor, n_clusters):
        super().__init__()
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(self, data):
        from sklearn.cluster import Birch

        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        birch = Birch(
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            n_clusters=self.n_clusters,
        )
        birch.fit(X)
        self.centroids = birch.subcluster_centers_
        return self

    def _transform(self, data):
        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(X, self.centroids)
        labels = np.argmin(distances, axis=1)
        rows = [Row(cluster=int(label)) for label in labels]
        return data.sql_ctx.createDataFrame(rows)


class HierarchicalCluster(Transformer):
    """
    Hierarchical clustering using NumPy.

    Hierarchical clustering creates a hierarchy of clusters either
    by repeatedly merging the closest pairs of clusters (agglomerative) or by splitting clusters (divisive).

    :param n_clusters: Number of clusters.
    :param linkage: Linkage method to use. Default is 'ward'.

    Example:

        from your_module import HierarchicalCluster

        hc = HierarchicalCluster(n_clusters=4, linkage='complete')
        model = hc.fit(data)
        clustered_data = model._transform(data)
    """

    def __init__(self, n_clusters, linkage="ward"):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None

    def fit(self, data):
        from scipy.cluster.hierarchy import linkage

        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        self.model = linkage(X, method=self.linkage)
        return self

    def _transform(self, data):
        from scipy.cluster.hierarchy import fcluster

        X = np.array(
            data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        )
        labels = fcluster(self.model, self.n_clusters, criterion="maxclust")
        rows = [Row(cluster=int(label)) for label in labels]
        return data.sql_ctx.createDataFrame(rows)


class HiddenMarkov(Transformer):
    """
    Hidden Markov Model.

    Hidden Markov Models (HMMs) are statistical models that represent
    systems that transition from one state to another, where the state is not
    directly visible but the output dependent on the state is visible.

    :param num_states: Number of states.
    :param num_observations: Number of observations.
    :param transition_probabilities: State transition probabilities.
    :param observation_probabilities: Observation probabilities.
    :param initial_probabilities: Initial state probabilities.

    Example:

        from your_module import HiddenMarkov

        hmm = HiddenMarkov(
            num_states=3,
            num_observations=2,
            transition_probabilities=[[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]],
            observation_probabilities=[[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]],
            initial_probabilities=[0.6, 0.3, 0.1]
        )
        path = hmm.viterbi([0, 1, 0])
    """

    def __init__(
        self,
        num_states,
        num_observations,
        transition_probabilities,
        observation_probabilities,
        initial_probabilities,
    ):
        super().__init__()
        self.num_states = num_states
        self.num_observations = num_observations
        self.transition_probabilities = transition_probabilities
        self.observation_probabilities = observation_probabilities
        self.initial_probabilities = initial_probabilities

    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.num_states))
        alpha[0, :] = (
            self.initial_probabilities
            * self.observation_probabilities[:, observations[0]]
        )
        for t in range(1, T):
            for j in range(self.num_states):
                alpha[t, j] = (
                    np.sum(alpha[t - 1, :] * self.transition_probabilities[:, j])
                    * self.observation_probabilities[j, observations[t]]
                )
        return alpha

    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.num_states))
        beta[-1, :] = 1
        for t in range(T - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = np.sum(
                    self.transition_probabilities[i, :]
                    * self.observation_probabilities[:, observations[t + 1]]
                    * beta[t + 1, :]
                )
        return beta

    def viterbi(self, observations):
        T = len(observations)
        delta = np.zeros((T, self.num_states))
        psi = np.zeros((T, self.num_states), dtype=int)
        delta[0, :] = (
            self.initial_probabilities
            * self.observation_probabilities[:, observations[0]]
        )
        for t in range(1, T):
            for j in range(self.num_states):
                max_val = delta[t - 1, :] * self.transition_probabilities[:, j]
                psi[t, j] = np.argmax(max_val)
                delta[t, j] = (
                    np.max(max_val) * self.observation_probabilities[j, observations[t]]
                )
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1, :])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path
