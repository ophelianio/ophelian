import numpy as np
import shap
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


class FeatureSelector(Transformer):
    """
    FeatureSelector class for performing feature selection using various techniques.

    The class supports several feature selection methods including chi-squared, mutual information,
    ANOVA F-value, Random Forest importance, Lasso, SHAP values, and deep learning-based selection.

    :param method: str, optional (default='chi2')
        The feature selection method to use. Supported methods are 'chi2', 'mutual_info',
        'f_classif', 'random_forest', 'lasso', 'shap', 'deep_learning'.
    :param k: int, optional (default=10)
        The number of top features to select.
    :param output_col: str, optional (default='selected_features')
        The name of the output column that will contain the selected features.

    Example:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("FeatureSelector").getOrCreate()
        data = [(1.0, 0.1, 0.3, 0.5, 0), (0.2, 0.4, 0.6, 0.8, 1), (0.5, 0.7, 0.9, 0.3, 0), (0.4, 0.9, 0.7, 0.6, 1)]
        columns = ["feature1", "feature2", "feature3", "feature4", "label"]
        df = spark.createDataFrame(data, columns)
        selector = FeatureSelector(method="shap", k=2)
        pipeline = Pipeline(stages=[selector])
        model = pipeline.fit(df)
        selected_df = model.transform(df)
        selected_df.show()
    """

    def __init__(self, method="chi2", k=10, output_col="selected_features"):
        super().__init__()
        self.method = method
        self.k = k
        self.output_col = output_col

    def _select_features(self, X, y):
        if self.method == "chi2":
            selector = SelectKBest(chi2, k=self.k).fit(X, y)
        elif self.method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=self.k).fit(X, y)
        elif self.method == "f_classif":
            selector = SelectKBest(f_classif, k=self.k).fit(X, y)
        elif self.method == "random_forest":
            model = RandomForestClassifier().fit(X, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[-self.k :]
            selector = np.zeros(X.shape[1], dtype=bool)
            selector[indices] = True
        elif self.method == "lasso":
            model = LassoCV().fit(X, y)
            selector = model.coef_ != 0
        elif self.method == "shap":
            model = RandomForestClassifier().fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_sum = np.abs(shap_values).mean(axis=0)
            indices = np.argsort(shap_sum)[-self.k :]
            selector = np.zeros(X.shape[1], dtype=bool)
            selector[indices] = True
        elif self.method == "deep_learning":
            input_layer = Input(shape=(X.shape[1],))
            dense_layer = Dense(64, activation="relu")(input_layer)
            output_layer = Dense(1, activation="sigmoid")(dense_layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer="adam", loss="binary_crossentropy")
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            importance = np.abs(model.layers[1].get_weights()[0]).sum(axis=1)
            indices = np.argsort(importance)[-self.k :]
            selector = np.zeros(X.shape[1], dtype=bool)
            selector[indices] = True
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")

        return selector

    def _transform(self, dataset: DataFrame) -> DataFrame:
        feature_cols = [col for col in dataset.columns if col != "label"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        assembled = assembler.transform(dataset)
        assembled_rdd = assembled.select("features", "label").rdd.map(
            lambda row: (row.features.toArray(), row.label)
        )

        X = np.array(assembled_rdd.map(lambda x: x[0]).collect())
        y = np.array(assembled_rdd.map(lambda x: x[1]).collect())

        selector = self._select_features(X, y)

        selected_features = [
            feature_cols[i] for i in range(len(feature_cols)) if selector[i]
        ]

        selected_df = dataset.select(selected_features + ["label"])

        return selected_df

    def _fit(self, dataset: DataFrame):
        return self

    def fit(self, dataset: DataFrame):
        return self._fit(dataset)

    def transform(self, dataset: DataFrame, params=None) -> DataFrame:
        return self._transform(dataset)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("FeatureSelector").getOrCreate()

    data = [
        (1.0, 0.1, 0.3, 0.5, 0),
        (0.2, 0.4, 0.6, 0.8, 1),
        (0.5, 0.7, 0.9, 0.3, 0),
        (0.4, 0.9, 0.7, 0.6, 1),
    ]

    columns = ["feature1", "feature2", "feature3", "feature4", "label"]
    df = spark.createDataFrame(data, columns)

    selector = FeatureSelector(method="shap", k=2)
    pipeline = Pipeline(stages=[selector])
    model = pipeline.fit(df)
    selected_df = model.transform(df)

    selected_df.show()
