from pyspark.sql import DataFrame

from ophelian._logger import OphelianLogger
from ophelian.ophelian_spark import PathWrite


class Write:

    def __init__(self):
        self.__logger = OphelianLogger()
        self.__path = PathWrite()

    def write_parquet(
        self,
        df: DataFrame,
        output_type: str,
        project: str,
        part: str = None,
        mode: str = None,
        rep: int = None,
    ) -> str:
        """
        Read file function helps to read multi-format file in spark
        :param df: spark DataFrame to write in parquet
        :param output_type: str, output name type for data-warehouse
        :param project: str, project name output data
        :param part: str, partition name field
        :param mode: str, spark writer mode
        :param rep: int, number of parts in each partition
        :return: spark DataFrame
        """
        if output_type not in [self.__path.model, self.__path.engine]:
            self.__logger.error(f"output '{output_type}' type not allowed")
            raise ValueError("you must choose 'engine' or 'model' for output type")
        else:
            path = self.__path.WritePath(output_type, project)
            mode = "overwrite" if not mode else mode
            rep = 5 if not rep else rep
            partition = [part]
            if not part:
                self.__logger.info("Write Parquet")
                df.coalesce(rep).write.mode(mode).parquet(path)
                self.__logger.info("Write Parquet Successfully")
                self.__logger.warning("Parquet Parts With No Partition...")
                self.__logger.info(f"New Parquet Path: {path}")
                return path
            self.__logger.info("Write Parquet")
            df.coalesce(rep).write.mode(mode).parquet(path, partitionBy=partition)
            self.__logger.info("Write Parquet Successfully")
            self.__logger.info(f"New Parquet Path: {path}{part}=yyy-MM-dd")
            return path

    def write_json(
        self,
        df: DataFrame,
        output_type: str,
        project: str,
        part: str = None,
        mode: str = None,
        rep: int = None,
    ) -> str:
        path = self.__path.WritePath(output_type, project)
        df.repartition(rep).write.mode(mode).partitionBy(part).json(path)
        self.__logger.info("Write Json")
        return path

    def write_csv(
        self,
        df: DataFrame,
        output_type: str,
        project: str,
        part: str = None,
        mode: str = None,
        rep: int = None,
    ) -> str:
        path = self.__path.WritePath(output_type, project)
        df.repartition(rep).write.mode(mode).partitionBy(part).csv(path)
        self.__logger.info("Write CSV")
        return path
