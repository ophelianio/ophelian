import random

import numpy as np
from pyspark.sql import DataFrame
from sklearn import neighbors

from ophelian.ophelian_spark import InstanceError
from ophelian.ophelian_spark.ml.feature_miner import (
    BuildVectorAssembler,
    NumpyToVector,
    SparkToNumpy,
)


class SyntheticSample:
    """
    Docstring class
    """

    @staticmethod
    def vector_assembling(df, target_name):
        """
        Vector assembling function will create a vector filled with features for each row

        :param df: df, spark DataFrame with target label
        :param target_name: str, string name from target label
        :return: DataFrame, table that includes the feature vector and label
        """
        InstanceError(df, DataFrame)
        InstanceError(target_name, str)
        column_names = list(df.drop(target_name).columns)
        vec_transform = BuildVectorAssembler(column_names).transform(df)
        return vec_transform.select(
            "features", (vec_transform[target_name]).alias("label")
        )

    @staticmethod
    def split_target(df, field, minor=1, major=0):
        """
        Split target will split in two distinct DataFrame from label '1' and '0'

        :param df: DataFrame, spark DataFrame with target label
        :param field: str, string name from target label
        :param minor: int, integer number for minority class; '1' set as default
        :param major: int, integer number for majority class; '0' set as default
        :return: dict, python dictionary with separated DataFrame
        """
        minor_df = df[df[field] == minor]
        major_df = df[df[field] == major]
        return {"minor": minor_df, "major": major_df}

    @staticmethod
    def spark_to_numpy(df, columns):
        """
        Spark to numpy function will help to dask_spark.py from spark DataFrame to numpy array
        in a distributed manner

        :param df: DataFrame, spark DataFrame with features column
        :param columns: list, with name of columns to matrix
        :return: np.array, numpy array object with feature elements
        """
        feature_df = BuildVectorAssembler(columns).transform(df).select("features")
        numpy_array = SparkToNumpy().transform(feature_df)
        return numpy_array

    @staticmethod
    def numpy_to_spark(feature_array, sc):
        """
        Numpy to spark function will help to dask_spark.py from numpy array to spark DataFrame
        in a distributed manner

        :param feature_array: np.array, numpy array object with feature elements
        :return: DataFrame, with features and label; 'features' and 'label' set as default
        """
        InstanceError(feature_array, np.ndarray)
        return NumpyToVector(sc).transform(feature_array)

    @staticmethod
    def __k_neighbor(k_n, algm, feature):
        """
        k neighbor will compute k-Nearest Neighbors sklearn algorithm

        :param k_n: int, integer number for k nearest neighbors groups; '2' set as default
        :param algm: str, string name for k-NN's algorithm choice; 'auto' set as default
        :param feature: np.array, np.array object with column features
        :return: list, python list with numpy array object for each neighbor
        """
        n_neighbor = neighbors.NearestNeighbors(n_neighbors=k_n, algorithm=algm)
        model_fit = n_neighbor.fit(feature)
        return model_fit.kneighbors(feature)

    @staticmethod
    def __compute_smo(neighbor_list, min_pct, min_arr):
        """
        Compute smo function will compute the SMOTE oversampling technique

        :param neighbor_list: list, python list with numpy array object for each neighbor
        :param min_pct: int, integer pct for over min; '100' set as default
        :param min_arr: list, python list with minority class rows
        :return: list, python list with sm class oversampled
        """
        smo = []
        counter = 0
        pct_over = int(min_pct / 100)
        while len(min_arr) > counter:
            for i in range(pct_over):
                random_neighbor = random.randint(0, len(neighbor_list) - 1)
                diff = neighbor_list[random_neighbor][0] - min_arr[i][0]
                new_record = min_arr[i][0] + random.random() * diff
                smo.insert(0, new_record)
            counter += 1
        return np.array(smo)

    @staticmethod
    def __build_split_df(df, label, sample):
        return SyntheticSample.split_target(df, label)[sample]

    @staticmethod
    def __build_neighbor_list(k, algrthm, feature_mat):
        return SyntheticSample.__k_neighbor(k, algrthm, feature_mat)[1]

    @staticmethod
    def __build_minor_target_array(df, column, spark):
        cache_df = df.cache()
        minor_target = cache_df.drop(column).rdd.map(lambda x: list(x)).collect()
        spark.catalog.clearCache()
        return minor_target

    @staticmethod
    def __build_synthetic_minority_over_sample(
        df, k, alg, pct, spark, label="label", features="features"
    ):
        data_min = SyntheticSample.__build_split_df(df, label, "minor")
        feat_mat = SyntheticSample.spark_to_numpy(data_min, features)
        neighbor = SyntheticSample.__build_neighbor_list(k, alg, feat_mat)
        min_array = SyntheticSample.__build_minor_target_array(data_min, label, spark)
        return SyntheticSample.__compute_smo(neighbor, pct, min_array)

    @staticmethod
    def __build_sample(df, smo_df, pct, label="label"):
        data_min = SyntheticSample.__build_split_df(df, label, "minor")
        data_max = SyntheticSample.__build_split_df(df, label, "major")
        smo_data_minor = data_min.union(smo_df)
        new_data_major = data_max.sample(False, (float(pct / 100)))
        return new_data_major.union(smo_data_minor)

    @staticmethod
    def smote_sampling(spark, df, k=2, alg="auto", pct_over_min=100, pct_under_max=100):
        """
        Smote sampling function will create an oversampling with SMOTE technique

        :param spark: spark session object
        :param df: DataFrame, spark DataFrame with features column
        :param k: int, integer k folds for k-NN's groups; '2' set as default
        :param alg: str, string name for k-NN's algorithm choice; 'auto' set as default
        :param pct_over_min: int, integer number for sampling minority class; '100' set as default
        :param pct_under_max: int, integer number for sampling majority class; '100' set as default
        :return: DataFrame, with new SMOTE features sampled
        """
        InstanceError(k, int)
        InstanceError(alg, str)
        InstanceError(pct_over_min, int)
        InstanceError(pct_under_max, int)
        if (pct_under_max < 10) | (pct_under_max > 100):
            raise ValueError(
                "value of variable 'pct_under_max' must be 10 <= pct <= 100"
            )
        if pct_over_min < 100:
            raise ValueError("value of variable 'min_pct' must be >= 100")
        if alg not in ("auto", "brute", "kd_tree", "ball_tree"):
            raise ValueError(f"unrecognized algorithm: '{alg}'")
        new_row = SyntheticSample.__build_synthetic_minority_over_sample(
            df, k, alg, pct_over_min, spark
        )
        smo_data_df = SyntheticSample.numpy_to_spark(feature_array=new_row, sc=spark)
        return SyntheticSample.__build_sample(df, smo_data_df, pct_under_max)
