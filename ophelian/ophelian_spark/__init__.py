from pyspark.sql.functions import coalesce, count, covar_pop, first, grouping
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import mean
from pyspark.sql.functions import min as spark_min
from pyspark.sql.functions import stddev
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import variance

"""Top-level package for ophelian_spark."""

__author__ = """LuisFalva"""
__email__ = "falvaluis@gmail.com"
__version__ = "0.1.3"
__all__ = [
    "SparkMethods",
    "ClassType",
    "ClassName",
    "InstanceError",
    "FormatRead",
    "PathWrite",
    "FormatWrite",
    "OphelianMLException",
    "OphelianMLMinerException",
    "OphelianUtilitiesException",
    "OphelianReadFileException",
    "OphelianSparkSessionException",
    "OphelianSparkWrapperException",
    "OphelianWriteFileException",
    "OphelianFunctionsException",
]


def SparkMethods():
    return {
        "sum": spark_sum,
        "min": spark_min,
        "max": spark_max,
        "mean": mean,
        "stddev": stddev,
        "var": variance,
        "first": first,
        "count": count,
        "coalesce": coalesce,
        "covar_pop": covar_pop,
        "grouping": grouping,
    }


def ClassType(dtype):
    return dtype.__class__


def ClassName(dtype):
    return dtype.__class__.__name__


def InstanceError(obj, t):
    if not isinstance(obj, t):
        raise TypeError(f"Unsupported Type {ClassName(obj)}")
    return None


class FormatRead:

    def __init__(self):
        self.parquet = "parquet"
        self.excel = "excel"
        self.csv = "csv"
        self.json = "json"
        self.all = [self.parquet, self.excel, self.csv, self.json]


class FormatWrite:
    def __init__(self):
        self.parquet = "parquet"
        self.excel = "excel"
        self.csv = "csv"
        self.json = "json"
        self.all = [self.parquet, self.excel, self.csv, self.json]


class PathWrite:
    def __init__(self):
        self.root = "data"
        self.dir = "ophelian_spark"
        self.out = "out"
        self.engine = "engine"
        self.model = "model"

    def WritePath(self, opt, project):
        return f"{self.root}/{self.dir}/{self.out}/{opt}/{project}/"


class OphelianMLException(Exception):
    """
    Ophelian ML Exception
    """

    pass


class OphelianMLMinerException(Exception):
    """
    Ophelian ML Miner Exception
    """

    pass


class OphelianReadFileException(Exception):
    """
    Ophelian Read File Exception
    """

    pass


class OphelianSparkSessionException(Exception):
    """
    Ophelian Spark Session Exception
    """

    pass


class OphelianUtilitiesException(Exception):
    """
    Ophelian Utilities Exception
    """

    pass


class OphelianSparkWrapperException(Exception):
    """
    Ophelian Spark Wrapper Exception
    """

    pass


class OphelianWriteFileException(Exception):
    """
    Ophelian Write File Exception
    """

    pass


class OphelianFunctionsException(Exception):
    """
    Ophelian Functions Exception
    """

    pass
