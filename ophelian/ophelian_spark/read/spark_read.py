from datetime import date

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from ophelian._logger import OphelianLogger
from ophelian.ophelian_spark import FormatRead, OphelianReadFileException
from ophelian.ophelian_spark.generic import union_all

__all__ = ["Read", "SparkReadWrapper"]


class Read:

    __logger = OphelianLogger()
    __format = FormatRead()

    @staticmethod
    def build_params(params) -> dict:
        path = params.getString("config.input.raw")
        stage = params.getString("config.stage")
        date = params.getString("config.target_date")
        source = params.getString("config.source")
        pivot = params.getString("config.pivot")
        return {
            "path": path,
            "stage": stage,
            "date": date,
            "source": source,
            "pivot": pivot,
        }

    @staticmethod
    def read_csv(
        self, path: str, header: bool = True, infer_schema: bool = True
    ) -> DataFrame:
        """
        Read csv function helps to read csv format file in spark
        :param self: sparkSession instance
        :param path: str, root path from file to load
        :param header: bool, True for header config
        :param infer_schema: bool, True for infer schema type
        :return: spark DataFrame
        """
        try:
            Read.__logger.info(f"Read CSV File From Path: {path}")
            return self.read.csv(path, header=header, inferSchema=infer_schema)
        except TypeError as te:
            raise OphelianReadFileException(
                f"An error occurred while calling read_csv() method: {te}"
            )

    @staticmethod
    def read_parquet(self, path: str) -> DataFrame:
        """
        Read parquet function helps to read parquet format file in spark
        :param self: sparkSession instance
        :param path: str, root path from file to load
        :return: spark DataFrame
        """
        try:
            Read.__logger.info(f"Read Parquet File From Path: {path}")
            return self.read.parquet(path)
        except TypeError as te:
            raise OphelianReadFileException(
                f"An error occurred while calling read_parquet() method: {te}"
            )

    @staticmethod
    def read_excel(self, path: str, sheet_name: str) -> DataFrame:
        """
        Read excel function helps to read excel format file in spark
        :param self: sparkSession instance
        :param path: str, root path from file to load
        :param sheet_name: str, name from excel sheet
        :return: spark DataFrame
        """
        try:
            Read.__logger.info(f"Read Excel File From Path: {path}")
            return self.createDataFrame(pd.read_excel(path, sheet_name))
        except TypeError as te:
            raise OphelianReadFileException(
                f"An error occurred while calling read_excel() method: {te}"
            )

    @staticmethod
    def read_json(self, path: str) -> DataFrame:
        """
        Read json function helps to read json format file in spark
        :param self: sparkSession instance
        :param path: str, root path from file to load
        :return: spark DataFrame
        """
        try:
            Read.__logger.info(f"Read Json File From Path: {path}")
            return self.read.json(path)
        except TypeError as te:
            raise OphelianReadFileException(
                f"An error occurred while calling read_json() method: {te}"
            )

    @staticmethod
    def read_file(
        self,
        path_source: str,
        source: str,
        sheet: str = None,
        header: bool = True,
        infer_schema: bool = True,
    ) -> DataFrame:
        """
        Read file function helps to read multi-format file in spark
        :param self: spark session instance
        :param path_source: str, root path from file to load
        :param source: str, source type format file
        :param sheet: str, name from excel sheet
        :param header: str, name from excel sheet
        :param infer_schema: str, name from excel sheet
        :return: spark DataFrame
        """
        try:
            if path_source is None or source is None:
                Read.__logger.error("'path' And 'source' Params Must Be Not None")
                raise ValueError("Params Must Be Not None")
            if source not in Read.__format.all:
                Read.__logger.error(f"Format Type '{source}' Not Available In Spark")
                raise ValueError(
                    "'source' Type Must Be: {'parquet', 'excel', 'csv', 'json'}"
                )
            read_object = {
                Read.__format.parquet: (
                    None
                    if source != "parquet"
                    else Read.read_parquet(self, path_source)
                ),
                Read.__format.csv: (
                    None
                    if source != "csv"
                    else Read.read_csv(self, path_source, header, infer_schema)
                ),
                Read.__format.excel: (
                    None
                    if source != "excel"
                    else Read.read_excel(self, path_source, sheet)
                ),
                Read.__format.json: (
                    None if source != "json" else Read.read_json(self, path_source)
                ),
            }
            return read_object[source]
        except ValueError as error:
            raise OphelianReadFileException(
                f"An error occurred while calling read_file() method: {error}"
            )

    @staticmethod
    def scan_update_data(
        self,
        path_pattern,
        partition: str = "partition_id",
        since_date: int = 2010,
        source: str = "parquet",
    ):
        """
        Scan update data helps to read the las available chunk of data.
        :param self: spark Session instance
        :param path_pattern: str, pattern path from loading directory
        :param partition: str, name of partition for each path, 'partition_id' set as default
        :param since_date: int, year from reading, '2010' set as default
        :param source: str, source type for reading file, 'parquet' set as default
        :return: spark DataFrame
        """
        try:
            now = date.today()
            scan_dates = list(range(since_date, now.year + 1))

            df_list = []
            non_paths = []
            for date_i in scan_dates[::-1]:
                scan_path = path_pattern + f"{partition}={date_i}"
                try:
                    scan = self.readFile(scan_path, source)
                    df_list.append(scan)
                except Exception:
                    Read.__logger.error(f"Path '{scan_path}' Does Not Exist.")
                    non_paths.append(scan_path.split("/")[-1])
                    continue
            Read.__logger.error(f"List Of Non Existing Elements. {non_paths}")

            df_union = union_all(df_list)
            self.catalog.clearCache()
            return {"df": df_union, "df_list": df_list, "non_exist": non_paths}
        except ValueError as error:
            raise OphelianReadFileException(
                f"An error occurred while calling scan_update_data() method: {error}"
            )


class SparkReadWrapper(SparkSession):

    SparkSession.readFile = Read.read_file
    SparkSession.scan_update = Read.scan_update_data
