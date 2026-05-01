from pyspark.sql import SparkSession

from ophelian._logger import OphelianLogger


class OphelianSpark:

    __logger = OphelianLogger()

    def __init__(self):
        self.ophelian_spark = None

    def __build_spark_app(self, app_name):
        self.ophelian_spark = SparkSession.builder.appName(app_name).getOrCreate()

    def __build_spark(self):
        self.ophelian_spark = SparkSession.builder.appName(
            "No 'appName' configured"
        ).getOrCreate()

    def __app_name(self):
        return self.ophelian_spark.sparkContext.appName

    def __spark_version(self):
        return self.ophelian_spark.version

    def __spark_ui_port(self):
        """
        Print spark UI port server for tracking Spark activity
        :return: spark ui port address
        """
        return self.ophelian_spark._sc.uiWebUrl

    def build_spark_session(self, app_name: str = None) -> SparkSession:
        """
        Build spark session is the builder method to initialize spark session under the hood of Ophelia
        :param app_name: str, app name description for spark job name
        :return: spark session
        """
        if not app_name:
            OphelianSpark.__logger.info("Build Spark Session")
            self.__build_spark()
            OphelianSpark.__logger.info("Spark Version: " + self.__spark_version())
            OphelianSpark.__logger.warning(
                "Please, Be Aware To Set App Name Next Time..."
            )
            OphelianSpark.__logger.info("Spark UI Address: '" + self.__spark_ui_port())
            return self.ophelian_spark
        else:
            OphelianSpark.__logger.warning("Initializing Spark Session")
            self.__build_spark_app(app_name=app_name)
            OphelianSpark.__logger.info("Spark Version: " + self.__spark_version())
            OphelianSpark.__logger.info("This Is: '" + self.__app_name() + "' App")
            OphelianSpark.__logger.info("Spark UI Address: " + self.__spark_ui_port())
            return self.ophelian_spark

    def build_spark_context(self):
        """
        Build spark context is the builder method to initialize spark context given a spark session
        :return: spark context
        """
        OphelianSpark.__logger.info("Spark Context Initialized Successfully")
        return self.ophelian_spark.sparkContext

    def clear_cache(self):
        self.ophelian_spark.catalog.clearCache()
        OphelianSpark.__logger.info("Clear Spark Cache Successfully")

    @staticmethod
    def ophelia_active_session():
        OphelianSpark.__logger.info("Ophelian Active Session")
        return SparkSession.getActiveSession()
