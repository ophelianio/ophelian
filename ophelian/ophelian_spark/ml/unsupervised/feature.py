import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.feature import PCA, PCAModel
from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.sql.types import Row

from ophelian._logger import OphelianLogger
from ophelian.ophelian_spark import OphelianMLException
from ophelian.ophelian_spark.ml.feature_miner import (
    BuildStandardScaler,
    BuildVectorAssembler,
)


class PCAnalysis(Transformer):
    """
    for PCA algorithm
    """

    def __init__(self, k=None, metadata_path=None):
        super().__init__()
        self.__k = k
        self.__metadata_path = metadata_path

    def __build_pca(self, df, metadata_path):
        pca = PCA(k=self.__k, inputCol="scaled_features", outputCol="pca_features")
        if self.__metadata_path:
            pca.fit(df).write().overwrite().save(metadata_path)
            return PCAModel.load(metadata_path).transform(df)
        return pca.fit(df).transform(df)

    def _transform(self, dataset):
        return self.__build_pca(dataset, self.__metadata_path)


class SingularVD(Transformer):
    """
    for SVD algorithm
    """

    def __init__(self, k=None, offset=95, label_col="label"):
        super().__init__()
        self.__logger = OphelianLogger()
        self.k = k
        self.offset = offset
        self.label_col = label_col

    def __to_dense_vector(self, vec_df):
        """

        :param vec_df:
        :return:
        """
        try:
            to_dense_udf = udf(lambda v: Vectors.dense(v.toArray()), VectorUDT())
            self.__logger.info("Parsing Feature Vector To DenseUDT")
            return vec_df.select(
                monotonically_increasing_id().alias("id"),
                to_dense_udf("features").alias("features"),
            )
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling __to_dense_vector() method: {te}"
            )

    def __standard_features(
        self,
        df,
        with_mean=True,
        with_std=True,
        model_path="data/master/ophelian_spark/out/model/save/StandardScalerModel",
    ):
        """
        Standardizes features by removing the mean and scaling to unit variance using column summary
        statistics on the samples in the training set and persisting the estimator into local root FS,
        in order to change this please set your own path. This also may work over NFS and HDFS as well.
        :param df: spark DataFrame to re-scale to a standard scale
        :param with_mean: bool to indicate compute mean corrected sample, True as default
        :param with_std: bool to indicate compute unit std corrected sample standard deviation, True as default
        :param model_path: path were estimator model will be saved
        :return: spark DataFrame with Standard Scaled features
        """
        try:
            vector_assemble_df = BuildVectorAssembler(
                df.drop(self.label_col).columns
            ).transform(df)
            dense_vec_df = self.__to_dense_vector(vector_assemble_df)
            self.__logger.info("Compute Feature Standard Scaler")
            if model_path is not None:
                return BuildStandardScaler(with_mean, with_std, model_path).transform(
                    dense_vec_df
                )
            return BuildStandardScaler(with_mean, with_std).transform(dense_vec_df)
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling __standard_features() method: {te}"
            )

    def __index_row_matrix_rdd(self, scale_df):
        """

        :param scale_df:
        :return:
        """
        try:
            vector_mllib = MLUtils.convertVectorColumnsFromML(
                scale_df, "scaled_features"
            ).drop("features")
            vector_rdd = vector_mllib.select("scaled_features", "id").rdd.map(
                lambda x: IndexedRow(x[1], x[0])
            )
            self.__logger.info("Build Index Row Matrix RDD")
            return IndexedRowMatrix(vector_rdd)
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling __index_row_matrix_rdd() method: {te}"
            )

    @staticmethod
    def find_k(var_array, offset=95):
        """

        :param var_array:
        :param offset:
        :return:
        """
        try:
            return np.argmax(var_array > offset) + 1
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling find_k() method: {te}"
            )

    def svd_set_dict(self, df, d, tot_row, k, compute_u=True):
        """

        :param df:
        :param d:
        :param tot_row:
        :param k:
        :param compute_u:
        :return:
        """
        try:
            if k is None:
                k = d
                self.__logger.warning(
                    f"Warning: Set automatic 'K' == d == {d} since parameter is not specified"
                )
            scaled_df = self.__standard_features(df)
            svd_rdd = self.__index_row_matrix_rdd(scaled_df).computeSVD(
                k=k, computeU=compute_u
            )
            self.__logger.warning(f"Compute SVD With K={k}, d={d}, n={tot_row}")
            return {
                "svd": svd_rdd,
                "U": svd_rdd.U,
                "S": svd_rdd.s.toArray(),
                "V": svd_rdd.V,
                "n": tot_row,
                "d": d,
            }
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling svd_set_dict() method: {te}"
            )

    def compute_eigenvalues(self, svd):
        """

        :param svd:
        :return:
        """
        try:
            self.__logger.info("Compute Eigenvalues & Eigenvectors From Hyperplane")
            return np.flipud(np.sort(svd["S"] ** 2 / (svd["n"] - 1)))
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling compute_eigenvalues() method: {te}"
            )

    def cumulative_variance(self, eigen_vals):
        """
        Cumulative variance method will compute the cumulative variance array foreach component,
        each element inside the array will describe the number of components to choose in order
        to get the desired cumulative variance explained from original data
        :param eigen_vals: numpy array with eigenvalues
        :return: numpy array with total variance for each component
        """
        try:
            self.__logger.info("Compute Variance For Each Component")
            return (eigen_vals.cumsum() / eigen_vals.sum()) * 100
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling cumulative_variance() method: {te}"
            )

    def __foreach_pc_var_print(self, search_func, tot_var, var_list=(75, 85, 95)):
        """

        :param search_func:
        :param tot_var:
        :param var_list:
        :return:
        """
        try:
            message = (
                lambda var: f"Components For {var}% Variance K={search_func(tot_var, var)}"
            )
            self.__logger.info(f"Cumulative Variance Array:\n{tot_var}")
            for v in var_list:
                self.__logger.warning(message(var=v))
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling __foreach_pc_var_print() method: {te}"
            )

    def optimal_pc(self, svd_set_dict, offset):
        """

        :param svd_set_dict:
        :param offset:
        :return:
        """
        try:
            tot_var = self.cumulative_variance(self.compute_eigenvalues(svd_set_dict))
            optimal_k = self.find_k(tot_var, offset=offset)
            self.__logger.info(f"Find K={optimal_k} Components For {offset}% Variance")
            self.__foreach_pc_var_print(self.find_k, tot_var)
            self.__logger.info(f"Fit Optimal PCA Model With K={optimal_k} Components")
            return PCAnalysis(
                optimal_k, "data/master/ophelian_spark/out/model/save/OptimalPCAModel"
            )
        except TypeError as te:
            raise OphelianMLException(
                f"An error occurred while calling optimal_pc() method: {te}"
            )

    def operate(self, df):
        svd_dict = self.svd_set_dict(
            df=df, d=len(df.columns), tot_row=df.count(), k=self.k
        )
        return self.optimal_pc(svd_set_dict=svd_dict, offset=self.offset)

    def _transform(self, dataset):
        return self.operate(df=dataset)


class IndependentComponent(Transformer):
    """
    for ICA algorithm
    """

    def __init__(self, n_components=None):
        super().__init__()
        self.__n_components = n_components
        self.__logger = OphelianLogger()

    @staticmethod
    def __center(X):
        return X - X.mean(axis=0)

    @staticmethod
    def __whiten(X):
        cov = np.cov(X, rowvar=False)
        U, S, _ = np.linalg.svd(cov)
        X_whiten = np.dot(X, np.dot(U, np.diag(1.0 / np.sqrt(S))))
        return X_whiten

    def __ica(self, X, iterations=1000, tol=1e-5):
        X = self.__center(X)
        X = self.__whiten(X)
        components = self.__n_components if self.__n_components else X.shape[1]
        W = np.random.randn(components, components)
        for i in range(iterations):
            W_old = W.copy()
            XW = np.dot(X, W.T)
            g = np.tanh(XW)
            g_prime = 1 - g**2
            W = W + np.dot(g.T, X) / X.shape[0] - np.dot(g_prime.mean(axis=0) * W.T, W)
            W = np.dot(np.linalg.inv(np.dot(W, W.T)) ** 0.5, W)
            if np.max(np.abs(np.abs(np.diag(np.dot(W, W_old.T))) - 1)) < tol:
                break
        return np.dot(W, X.T).T

    def _transform(self, dataset):
        try:
            np_data = np.array(
                dataset.select("features").rdd.map(lambda x: x[0].toArray()).collect()
            )
            transformed_data = self.__ica(np_data)
            rows = [Row(ica_features=DenseVector(row)) for row in transformed_data]
            return dataset.sql_ctx.createDataFrame(rows)
        except Exception as e:
            self.__logger.error(f"An error occurred in ICA transform: {e}")
            raise OphelianMLException(f"An error occurred in ICA transform: {e}")


class LinearDAnalysis(Transformer):
    """
    for Linear Discriminant Analysis algorithm
    """

    def __init__(self, n_components=None):
        super().__init__()
        self.__n_components = n_components
        self.__logger = OphelianLogger()

    def __lda(self, X, y):
        class_labels = np.unique(y)
        mean_vectors = []
        for cl in class_labels:
            mean_vectors.append(np.mean(X[y == cl], axis=0))
        overall_mean = np.mean(X, axis=0)
        S_W = np.zeros((X.shape[1], X.shape[1]))
        for cl, mv in zip(class_labels, mean_vectors):
            class_scatter = np.cov(X[y == cl].T)
            S_W += class_scatter
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == class_labels[i], :].shape[0]
            mean_vec = mean_vec.reshape(X.shape[1], 1)
            overall_mean = overall_mean.reshape(X.shape[1], 1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        eig_pairs = [
            (np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))
        ]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        W = np.hstack(
            [eig_pairs[i][1].reshape(X.shape[1], 1) for i in range(self.__n_components)]
        )
        return X.dot(W)

    def _transform(self, dataset):
        try:
            np_data = np.array(
                dataset.select("features").rdd.map(lambda x: x[0].toArray()).collect()
            )
            labels = np.array(dataset.select("label").rdd.map(lambda x: x[0]).collect())
            transformed_data = self.__lda(np_data, labels)
            rows = [Row(lda_features=DenseVector(row)) for row in transformed_data]
            return dataset.sql_ctx.createDataFrame(rows)
        except Exception as e:
            self.__logger.error(f"An error occurred in LDA transform: {e}")
            raise OphelianMLException(f"An error occurred in LDA transform: {e}")


class LLinearEmbedding(Transformer):
    """
    for Locally Linear Embedding Analysis algorithm
    """

    def __init__(self, n_neighbors=5, n_components=2):
        super().__init__()
        self.__n_neighbors = n_neighbors
        self.__n_components = n_components
        self.__logger = OphelianLogger()

    def __lle(self, X):
        from sklearn.neighbors import NearestNeighbors

        N = X.shape[0]
        neighbors = NearestNeighbors(n_neighbors=self.__n_neighbors).fit(X)
        W = np.zeros((N, N))
        for i in range(N):
            Z = X[neighbors.kneighbors([X[i]], return_distance=False)[0][1:]]
            C = np.dot(Z - X[i], (Z - X[i]).T)
            W[i, neighbors.kneighbors([X[i]], return_distance=False)[0][1:]] = (
                np.linalg.solve(C, np.ones(self.__n_neighbors - 1))
            )
            W[i, neighbors.kneighbors([X[i]], return_distance=False)[0][1:]] /= np.sum(
                W[i, neighbors.kneighbors([X[i]], return_distance=False)[0][1:]]
            )
        M = (np.eye(N) - W).T.dot(np.eye(N) - W)
        eig_vals, eig_vecs = np.linalg.eig(M)
        indices = np.argsort(eig_vals)[1 : self.__n_components + 1]
        return eig_vecs[:, indices]

    def _transform(self, dataset):
        try:
            np_data = np.array(
                dataset.select("features").rdd.map(lambda x: x[0].toArray()).collect()
            )
            transformed_data = self.__lle(np_data)
            rows = [Row(lle_features=DenseVector(row)) for row in transformed_data]
            return dataset.sql_ctx.createDataFrame(rows)
        except Exception as e:
            self.__logger.error(f"An error occurred in LLE transform: {e}")
            raise OphelianMLException(f"An error occurred in LLE transform: {e}")


class StochasticNeighbor(Transformer):
    """
    for t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm
    """

    def __init__(
        self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000
    ):
        super().__init__()
        self.__n_components = n_components
        self.__perplexity = perplexity
        self.__learning_rate = learning_rate
        self.__n_iter = n_iter
        self.__logger = OphelianLogger()

    def __tsne(self, X):
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=self.__n_components,
            perplexity=self.__perplexity,
            learning_rate=self.__learning_rate,
            n_iter=self.__n_iter,
        )
        return tsne.fit_transform(X)

    def _transform(self, dataset):
        try:
            np_data = np.array(
                dataset.select("features").rdd.map(lambda x: x[0].toArray()).collect()
            )
            transformed_data = self.__tsne(np_data)
            rows = [Row(tsne_features=DenseVector(row)) for row in transformed_data]
            return dataset.sql_ctx.createDataFrame(rows)
        except Exception as e:
            self.__logger.error(f"An error occurred in t-SNE transform: {e}")
            raise OphelianMLException(f"An error occurred in t-SNE transform: {e}")
