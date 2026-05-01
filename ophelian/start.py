class OphelianSession:

    def __init__(self, app_name=None, no_mask=False):
        from ophelian._info import OphelianInfo
        from ophelian.ophelian_spark.functions import (
            CorrMatWrapper,
            CrossTabularWrapper,
            DaskSparkWrapper,
            DynamicSamplingWrapper,
            EfficientFrontierRatioCalculatorWrapper,
            JoinsWrapper,
            MapItemWrapper,
            NullDebugWrapper,
            PctChangeWrapper,
            ReshapeWrapper,
            RiskParityCalculatorWrapper,
            RollingWrapper,
            SelectsWrapper,
            ShapeWrapper,
            SharpeRatioCalculatorWrapper,
            SortinoRatioCalculatorWrapper,
        )
        from ophelian.ophelian_spark.ml.feature_miner import (
            BuildOneHotEncoder,
            BuildStandardScaler,
            BuildStringIndex,
            BuildVectorAssembler,
            NumpyToVector,
            SparkToNumpy,
        )
        from ophelian.ophelian_spark.ml.sampling.synthetic_sample import SyntheticSample
        from ophelian.ophelian_spark.ml.unsupervised.feature import (
            PCAnalysis,
            SingularVD,
        )
        from ophelian.ophelian_spark.read.spark_read import Read
        from ophelian.ophelian_spark.session.spark import OphelianSpark
        from ophelian.ophelian_spark.write.spark_write import Write

        transformers = {
            "spark_write": Write,
            "spark_read": Read,
            "ophelian_spark": OphelianSpark,
            # Spark wrapper functions class
            "null_debug": NullDebugWrapper,
            "correlation": CorrMatWrapper,
            "shape": ShapeWrapper,
            "rolling": RollingWrapper,
            "sampling": DynamicSamplingWrapper,
            "selects": SelectsWrapper,
            "map_item": MapItemWrapper,
            "reshape": ReshapeWrapper,
            "pct_change": PctChangeWrapper,
            "crosstab": CrossTabularWrapper,
            "joins": JoinsWrapper,
            "dask_spark": DaskSparkWrapper,
            "sortino_calculator": SortinoRatioCalculatorWrapper,
            "sharpe_calculator": SharpeRatioCalculatorWrapper,
            "efficient_frontier": EfficientFrontierRatioCalculatorWrapper,
            "risk_parity": RiskParityCalculatorWrapper,
            # Feature miner class
            "standard_scaler": BuildStandardScaler,
            "vector_assembler": BuildVectorAssembler,
            "string_index": BuildStringIndex,
            "one_hot_encode": BuildOneHotEncoder,
            # Spark ML pipelines
            "numpy_to_vector": NumpyToVector,
            "spark_to_numpy": SparkToNumpy,
            "pca_analysis": PCAnalysis,
            "singular_vd": SingularVD,
            "smote_sample": SyntheticSample,
        }
        OphelianInfo().print_info(no_mask)
        self.Spark = OphelianSpark()
        self.SparkSession = self.Spark.build_spark_session(app_name)
        self.SC = self.Spark.build_spark_context()
