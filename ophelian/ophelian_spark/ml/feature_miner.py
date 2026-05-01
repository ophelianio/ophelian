from math import log

import numpy as np
from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import (
    OneHotEncoder,
    OneHotEncoderModel,
    StandardScaler,
    StandardScalerModel,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import Row

from ophelian._logger import OphelianLogger
from ophelian.ophelian_spark import OphelianMLMinerException


class BuildStringIndex(Transformer):
    """
    Build String Index object will compute a Spark DataFrame with string column indexing to numeric codes
    codification string will map 1:1 unique string to a unique code number. By default, this is ordered
    by label frequencies so the most frequent label gets index 0, the less one gets index 1 and so on so forth.

    Note: Specify 'estimator_path' will require to set a directory name, this parameter will create a metadata
    model version on disk (e.g. hdfs). Also this helps to reduce memory usage on the train and predict stage.

    :param input_cols: string or list of string column names to index, a.k.a. categorical data type
    :param path: disk path to persist metadata model estimator. Optional.
    :param dir_name: directory name for persist metadata model estimator inside 'path'. Optional.

    Example:

        from ophelian_spark.ml.feature_miner import BuildStringIndex
        df = spark.createDataFrame(
            [('apple','red'), ('banana','yellow'), ('coconut','brown')],
            ["fruit_type", "fruit_color"]
        )
        string_cols_list = ['fruit_type', 'fruit_color']
        indexer = BuildStringIndex(string_cols_list, '/path/estimator/save/metadata/', 'StringIndex')
        indexer.transform(df).show(5, False)
    """

    def __init__(self, input_cols, path=None, dir_name=None):
        super().__init__()
        self.__logger = OphelianLogger()
        self.__input_cols = input_cols
        self.__dir_name = dir_name
        self.__estimator_path = path

    def __single_string_indexer(self, single_col):
        """
        Single string indexer method creates 'StringIndexer' pyspark.ml object
        :param single_col: str column name from str datatype column to index
        :return: StringIndexer
        """
        try:
            self.__logger.info("Creating Single String Indexers")
            return StringIndexer(inputCol=single_col, outputCol=f"{single_col}_index")
        except TypeError as te:
            raise OphelianMLMinerException(
                f"An error occurred while calling single_string_indexer() method: {te}"
            )

    def __multi_string_indexer(self, multi_col):
        """
        Multi string indexer method creates a list of 'StringIndexer'
        pyspark.ml object by each given column
        :param multi_col: list for column string names to index
        :return: list of multi StringIndexer
        """
        try:
            self.__logger.info("Creating Multi String Indexers")
            return [
                StringIndexer(inputCol=column, outputCol=f"{column}_index")
                for column in multi_col
            ]
        except TypeError as te:
            raise OphelianMLMinerException(
                f"An error occurred while calling multi_string_indexer() method: {te}"
            )

    def __build_string_indexer(self, df, col_name, dir_name):
        """
        String Indexer builder wrapper for single and multi string indexing
        :param df: Spark DataFrame object auto-reference from DataFrame class
        :param col_name: any column(s) name(s)
        :param dir_name: Name for new directory for persist model metadata estimator
        :return: transformed spark DataFrame
        """
        try:
            if isinstance(col_name, list):
                pipe_model = Pipeline(
                    stages=[*self.__multi_string_indexer(multi_col=col_name)]
                )
                if self.__estimator_path:
                    estimator_path = self.__estimator_path + dir_name
                    self.__logger.info(f"Estimator Metadata Path: {estimator_path}")
                    pipe_model.fit(df).write().overwrite().save(estimator_path)
                    self.__logger.info(
                        "Loading Multi String Indexer Estimator For Prediction"
                    )
                    return PipelineModel.load(estimator_path).transform(df)
                else:
                    self.__logger.info("Compute Multi String Indexer")
                    return pipe_model.fit(df).transform(df)
            else:
                pipe_model = Pipeline(
                    stages=[self.__single_string_indexer(single_col=col_name)]
                )
                if self.__estimator_path:
                    estimator_path = self.__estimator_path + dir_name
                    self.__logger.info(f"Estimator Metadata Path: {estimator_path}")
                    pipe_model.fit(df).write().overwrite().save(estimator_path)
                    self.__logger.info(
                        "Loading Single String Indexer Estimator For Prediction"
                    )
                    return PipelineModel.load(estimator_path).transform(df)
                else:
                    self.__logger.info("Compute Single String Indexer")
                    return pipe_model.fit(df).transform(df)
        except TypeError as te:
            raise OphelianMLMinerException(
                f"An error occurred while calling __build_string_indexer() method: {te}"
            )

    def _transform(self, dataset):
        return self.__build_string_indexer(dataset, self.__input_cols, self.__dir_name)


class BuildOneHotEncoder(Transformer):
    """
    Build One Hot Encoder class will build a SPark DataFrame One Hot Encoder Estimator,
    this is different from sklearn. This will map a previously indexed category column
    to a binary vector, foe each string index this will compute a unique binary vector.

    Note: this can handle invalid categories (such as typos or error types) discarding them,
    set handle_invalid = 'keep' to keep error so this will encode invalid values to all-zero vector.

    :param input_cols: string or list of index column names to encode
    :param path: disk path to persist metadata model estimator. Optional.
    :param dir_name: directory name for persist metadata model estimator inside 'path'. Optional.
    :param drop_last: If True creates a dummy encoding removing the last binary category for an all-zero vector
    :param handle_invalid: Set by default 'error' to discard error typos or error types from categorical columns

    Example:

        from ophelian_spark.ml.feature_miner import BuildOneHotEncoder
        df = spark.createDataFrame(
            [('0.0','0.2'), ('0.1','0.0'), ('0.2','0.1')],
            ["fruit_type_index", "fruit_color_index"]
        )
        indexed_cols_list = ['fruit_type_index', 'fruit_color_index']
        encoder = BuildOneHotEncoder(indexed_cols_list, '/path/estimator/save/metadata/', 'OneHotEncoder')
        encoder.transform(df).show(5, False)

    """

    def __init__(
        self,
        input_cols,
        path=None,
        dir_name=None,
        indexer=False,
        drop_last=True,
        handle_invalid="error",
    ):
        super().__init__()
        self.__logger = OphelianLogger()
        self.__input_cols = input_cols
        self.__estimator_path = path
        self.__dir_name = dir_name
        self.__indexer = indexer
        self.__drop_last = drop_last
        self.__handle_invalid = handle_invalid

    def __ohe_estimator(self, indexer_cols):
        """
        One hot encoder estimator method creates 'OneHotEncoder' pyspark.ml object
        :param indexer_cols: list for column string names to encode
        :return: OneHotEncoder transformer object
        """
        try:
            self.__logger.info("Creating Feature Encoders")
            return OneHotEncoder(
                inputCols=[indexer.getOutputCol() for indexer in indexer_cols],
                outputCols=[
                    f"{indexer.getOutputCol()}_encoded" for indexer in indexer_cols
                ],
                dropLast=self.__drop_last,
                handleInvalid=self.__handle_invalid,
            )
        except TypeError as te:
            raise OphelianMLMinerException(
                f"An error occurred while calling ohe_estimator() method: {te}"
            )

    def __build_one_hot_encoder(self, df, col_name, dir_name, indexer):
        """
        One hot encoder builder wrapper for encoding categorical features.
        Caution, too many levels on factors could derive in sparsity vectors.
        :param df: Spark DataFrame object auto-reference from DataFrame class
        :param col_name: Column previously indexed names to encode
        :param dir_name: Name for new directory for persist model metadata estimator
        :param indexer: Boolean flag to activate auto-indexer cols
        :return: One Hot Encoder Estimator Spark DataFrame
        """
        try:
            if indexer:
                indexers = BuildStringIndex(col_name).transform(df)
                encoder = Pipeline(stages=[self.__ohe_estimator(indexers)])
            else:
                encoder = Pipeline(stages=[self.__ohe_estimator(col_name)])

            if self.__estimator_path and dir_name:
                estimator_path = self.__estimator_path + dir_name
                self.__logger.warning(f"Estimator Metadata Path: {estimator_path}")
                encoder.fit(df).write().overwrite().save(estimator_path)
                self.__logger.info("Loading Encoder Estimator For Prediction")
                return OneHotEncoderModel.load(estimator_path).transform(df)

            encoder = Pipeline(stages=[self.__ohe_estimator(col_name)])
            self.__logger.info("Compute One Hot Encoder Estimator DataFrame")
            return encoder.fit(df).transform(df)
        except TypeError as te:
            raise OphelianMLMinerException(
                f"An error occurred while calling __build_one_hot_encoder() method: {te}"
            )

    def _transform(self, dataset):
        return self.__build_one_hot_encoder(
            dataset, self.__input_cols, self.__dir_name, self.__indexer
        )


class BuildVectorAssembler(Transformer):

    def __init__(self, input_cols, name_vec="features"):
        super().__init__()
        self.__logger = OphelianLogger()
        self.__input_cols = input_cols
        self.__name_vec = name_vec

    def __build_vector_assembler(self, df, input_cols, name_vec_col):
        """
        Vector assembler builder wrapper for sparse and dense vectorization, only numeric features accepted
        :param df: Spark DataFrame object auto-reference from DataFrame class
        :param input_cols: input name columns, 'string' for single feature, list for multiple feature
        :param name_vec_col: 'features' by default. Set new name for vector transformation column
        :return: Vector Assembler Transformation
        """
        try:
            vec_assembler = VectorAssembler(
                inputCols=input_cols, outputCol=name_vec_col
            )
            self.__logger.info("Build Vector Assembler DataFrame")
            return vec_assembler.transform(df)
        except TypeError as te:
            raise OphelianMLMinerException(
                f"An error occurred while calling build_vector_assembler() method: {te}"
            )

    def _transform(self, dataset):
        return self.__build_vector_assembler(
            dataset, self.__input_cols, self.__name_vec
        )


class BuildStandardScaler(Transformer):
    """ """

    def __init__(
        self,
        with_mean=False,
        with_std=True,
        path=None,
        input_col="features",
        output_col="scaled_features",
    ):
        super().__init__()
        self.__logger = OphelianLogger()
        self.__with_mean = with_mean
        self.__with_std = with_std
        self.__path = path
        self.__input_col = input_col
        self.__output_col = output_col

    def build_standard_scaler(
        self,
        df,
        with_mean=False,
        with_std=True,
        persist_estimator_path=None,
        input_col="features",
        output_col="scaled_features",
    ):
        """
        Standard Scaler estimator builder and transformer for dense feature vectors.
        Warnings: It will build a dense output, so take care when applying to sparse input.
        :param df: Spark DataFrame object auto-reference from DataFrame class
        :param with_mean: False by default. Centers the data with mean before scaling
        :param with_std: True by default. Scales the data to unit standard deviation
        :param persist_estimator_path: Persist model estimator metadata path
        :param input_col: Name for input column to scale
        :param output_col: Name of output column to create with scaled features
        :return: Standard Scaler model
        """
        std_scaler = StandardScaler(
            withMean=with_mean,
            withStd=with_std,
            inputCol=input_col,
            outputCol=output_col,
        )
        if persist_estimator_path:
            self.__logger.info("Compute Feature Standard ScalerModel Metadata")
            self.__logger.warning(
                f"Persist Metadata Model Path: {persist_estimator_path}"
            )
            std_scaler.fit(df).write().overwrite().save(persist_estimator_path)
            self.__logger.info("Loading Scaler Estimator For Prediction")
            return StandardScalerModel.load(persist_estimator_path).transform(df)
        self.__logger.info("Compute Feature Standard Scaler DataFrame")
        return std_scaler.fit(df).transform(df)

    def _transform(self, dataset):
        return self.build_standard_scaler(
            dataset,
            self.__with_mean,
            self.__with_std,
            self.__path,
            self.__input_col,
            self.__output_col,
        )


class SparkToNumpy(Transformer):
    """ """

    def __init__(self, list_columns=None):
        super().__init__()
        self.__logger = OphelianLogger()
        self.__list_columns = list_columns

    def __spark_to_numpy(self, df, columns=None):
        """
        Spark to numpy converter from features of the DataFrame
        :param df: Spark DataFrame object auto-reference from DataFrame class
        :param columns: list columns string name, optional
        :return: np.array with features
        """
        if columns is None:
            np_numbers = ["float", "double", "decimal", "integer"]
            columns = [k for k, v in df.dtypes if v in np_numbers]
        feature_df = (
            BuildVectorAssembler(columns).transform(df).select("features").cache()
        )
        to_numpy = np.asarray(feature_df.rdd.map(lambda x: x[0]).collect())
        feature_df.unpersist()
        self.__logger.info("Spark to Numpy Converter")
        return to_numpy

    def _transform(self, dataset):
        return self.__spark_to_numpy(dataset, self.__list_columns)


class NumpyToVector:
    """ """

    def __init__(self, SparkContext):
        super().__init__()
        self._sc = SparkContext
        self.__logger = OphelianLogger()

    def __numpy_to_vector_assembler(self, np_object, label_t=1):
        """
        Numpy to spark vector converter from a numpy object
        :param np_object: numpy array with features
        :param label_t: label type column, 1 as default
        :return: build from np.array to spark DataFrame
        """
        data_set = self._sc.parallelize(np_object)
        data_rdd = data_set.map(lambda x: (Row(features=DenseVector(x), label=label_t)))
        self.__logger.info("Numpy to Spark Converter")
        return data_rdd.toDF()

    def transform(self, np_object):
        return self.__numpy_to_vector_assembler(np_object)

    @staticmethod
    def probability_class(node):
        node_sum = sum(node.values())
        percents = {c: v / node_sum for c, v in node.items()}
        return node_sum, percents

    def gini_score(self, node):
        """
        Gini score function will compute the Gini index from a
        :param node:
        :return:
        """
        _, percents = self.probability_class(node)
        score = round(1 - sum([i**2 for i in percents.values()]), 3)
        self.__logger.info(f"Gini Score for node {node}: {score}")
        return score

    def entropy_score(self, node):
        _, percents = self.probability_class(node)
        score = round(sum([-i * log(i, 2) for i in percents.values()]), 3)
        self.__logger.info(f"Entropy Score for node {node}: {score}")
        return score

    def information_gain(self, parent, children, criterion):
        score = {"gini": self.gini_score, "entropy": self.entropy_score}
        metric = score[criterion](parent)
        parent_score = metric
        parent_sum = sum(parent.values())
        weighted_child_score = sum(
            [metric(i) * sum(i.values()) / parent_sum for i in children]
        )
        gain = round((parent_score - weighted_child_score), 2)
        self.__logger.info(f"Information gain: {gain}")
        return gain
