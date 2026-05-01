import os
import random
import re

import numpy as np
import pandas as pd
from dask import array as dask_arr
from dask import dataframe as dask_df
from py4j.protocol import Py4JJavaError
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation as SparkCorrelation
from pyspark.sql import DataFrame, SparkSession, SQLContext, Window
from pyspark.sql.column import _to_seq
from pyspark.sql.functions import (
    array,
    broadcast,
    col,
    concat_ws,
    count,
    explode,
    expr,
    isnan,
    isnull,
    lag,
    lit,
    mean,
    monotonically_increasing_id,
)
from pyspark.sql.functions import round as spark_round
from pyspark.sql.functions import row_number, stddev, struct, variance, when
from pyspark.sql.types import StringType, StructField, StructType
from quadprog import solve_qp

from ophelian.ophelian_spark import OphelianFunctionsException, SparkMethods
from ophelian.ophelian_spark._wrapper import DataFrameWrapper
from ophelian.ophelian_spark.generic import (
    feature_pick,
    regex_expr,
    remove_duplicate_element,
    union_all,
)
from ophelian.ophelian_spark.ml.optim.utils import LBFGS
from ophelian.ophelian_spark.session.spark import OphelianSpark

__all__ = [
    "NullDebugWrapper",
    "CorrMatWrapper",
    "ShapeWrapper",
    "MapItemWrapper",
    "RollingWrapper",
    "DynamicSamplingWrapper",
    "SelectsWrapper",
    "ReshapeWrapper",
    "PctChangeWrapper",
    "CrossTabularWrapper",
    "JoinsWrapper",
    "DaskSparkWrapper",
    "SortinoRatioCalculatorWrapper",
    "SharpeRatioCalculatorWrapper",
    "EfficientFrontierRatioCalculatorWrapper",
    "RiskParityCalculatorWrapper",
]


def _wrapper(wrap_object):
    _wrap = DataFrameWrapper()
    if isinstance(wrap_object, tuple):
        for wo in wrap_object:
            _wrap(wrap_object=wo)
    else:
        _wrap(wrap_object=wrap_object)


class NullDebug:

    @staticmethod
    def __cleansing_list(df, partition_by=None, offset: float = 0.5):
        if partition_by is None:
            raise TypeError(
                f"'partition_by' required parameter, invalid {partition_by} input."
            )
        cache_transpose = df.cache()
        null_counts = cache_transpose.groupBy(partition_by).agg(
            count(when(isnan(col("values")) | isnull(col("values")), "values")).alias(
                "null_count"
            ),
            count(lit(1)).alias("total_count"),
        )
        null_counts = null_counts.withColumn(
            "null_pct", col("null_count") / col("total_count")
        )
        valid_partitions = null_counts.filter(col("null_pct") <= offset).select(
            partition_by
        )
        return valid_partitions

    @staticmethod
    def null_clean(df, partition_by, offset):
        try:
            if partition_by:
                cleansing_list = NullDebug.__cleansing_list(df, partition_by, offset)
                return df.join(cleansing_list, on=partition_by, how="inner")
            return df
        except Exception as error:
            raise Exception(f"An error occurred on null_clean() method: {error}")


class NullDebugWrapper:
    """
    Class NullDebugWrapper is a class for wrapping methods from NullDebug class
    adding this functionality to Spark DataFrame class
    """

    _wrapper(wrap_object=NullDebug.null_clean)


class CorrMat:

    __spark = OphelianSpark().ophelia_active_session()

    @staticmethod
    def __corr(pair):
        """
        Corr private method distribute pandas series operations for correlation between RDD
        :param pair: RDD pair of elements
        :return: product of correlation matrix
        """
        (prod1, series1), (prod2, series2) = pair
        corr = pd.Series(series1).corr(pd.Series(series2))
        return prod1, prod2, float(corr)

    @staticmethod
    def __build_corr_label(mtd):
        min_level = 0.1
        mid_level = 0.3
        max_level = 0.5
        return when(col(f"{mtd}_coeff") < min_level, lit("very_low")).otherwise(
            when(
                (col(f"{mtd}_coeff") < min_level) & (col(f"{mtd}_coeff") <= mid_level),
                lit("low"),
            ).otherwise(
                when(
                    (col(f"{mtd}_coeff") < mid_level)
                    & (col(f"{mtd}_coeff") <= max_level),
                    lit("mid"),
                ).otherwise(when(col(f"{mtd}_coeff") > max_level, lit("high")))
            )
        )

    @staticmethod
    def cartesian_rdd(self, default_pivot, rep=20):
        if default_pivot is None:
            raise ValueError("'default_pivot' must be specified")
        rep_df = self.repartition(rep)
        numerical_cols = rep_df.columns[1:]
        to_wide_rdd = rep_df.rdd.map(
            lambda x: (x[default_pivot], [x[c] for c in numerical_cols])
        )
        return to_wide_rdd.cartesian(to_wide_rdd)

    @staticmethod
    def correlation_matrix(self, pivot_col=None, method="pearson", offset=0.7, rep=20):
        default_pivot = pivot_col if pivot_col is not None else self.columns[0]
        cartesian_rdd = CorrMat.cartesian_rdd(
            self, default_pivot=default_pivot, rep=rep
        )
        new_col_list = [
            f"{default_pivot}_m_dim",
            f"{default_pivot}_n_dim",
            f"{method}_coeff",
        ]
        corr_df = cartesian_rdd.map(CorrMat.__corr).toDF(schema=new_col_list)
        offset_condition = when(col(f"{method}_coeff") >= offset, lit(1.0)).otherwise(
            0.0
        )
        corr_label = CorrMat.__build_corr_label(method)
        return corr_df.select(
            "*", offset_condition.alias("offset"), corr_label.alias(f"{method}_label")
        )

    @staticmethod
    def unique_row(self, col):
        categories_rows = self.select(col).groupBy(col).count().toPandas()
        return sorted(categories_rows[col].unique())

    @staticmethod
    def vector_assembler(self, cols_name):
        vec_assembler = VectorAssembler(inputCols=cols_name, outputCol="features")
        return vec_assembler.transform(self)

    @staticmethod
    def corr_test(self, group_by, pivot_col, agg_dict, method="pearson"):
        matrix_df = self.toWide(
            group_by=group_by, pivot_col=pivot_col, agg_dict=agg_dict
        ).cache()
        vec_df = CorrMat.vector_assembler(
            self=matrix_df, cols_name=matrix_df.columns[1:]
        )
        return SparkCorrelation.corr(vec_df, "features", method).select(
            col(f"{method}(features)").alias(f"{method}_features")
        )

    @staticmethod
    def spark_correlation(self, group_by, pivot_col, agg_dict, method="pearson"):
        try:
            categories_list = self.uniqueRow(pivot_col)
            corr_test = CorrMat.corr_test(
                self=self,
                group_by=group_by,
                pivot_col=pivot_col,
                agg_dict=agg_dict,
                method=method,
            ).cache()
            corr_cols = [
                f"{method}_{c}_{b}" for c in categories_list for b in categories_list
            ]
            rdd_map = corr_test.rdd.map(
                lambda row: tuple(float(x) for x in row[f"{method}_features"].values)
            )
            CorrMat.__spark.catalog.clearCache()
            return rdd_map.toDF(corr_cols)
        except Exception as error:
            raise OphelianFunctionsException(
                f"An error occurred on spark_correlation() method: {error}"
            )


class CorrMatWrapper:
    """
    Class CorrMatWrapper is a class for wrapping methods from CorrMat class
    adding this functionality to Spark DataFrame class
    """

    func = (
        CorrMat.unique_row,
        CorrMat.cartesian_rdd,
        CorrMat.correlation_matrix,
        CorrMat.vector_assembler,
        CorrMat.spark_correlation,
    )
    _wrapper(wrap_object=func)


class Shape:

    @staticmethod
    def shape(self):
        if len(self.columns) == 1:
            return (self.count(),)
        return self.count(), len(self.columns)


class ShapeWrapper:
    """
    Class ShapeWrapper is a class for wrapping methods from Shape class
    adding this functionality to Spark DataFrame class
    """

    DataFrame.Shape = property(lambda self: Shape.shape(self))


class Rolling:

    __spark_methods = SparkMethods()

    @staticmethod
    def rolling_down(self, op_col, nat_order, min_p=2, window=2, method="sum"):
        w = Window.orderBy(nat_order).rowsBetween(
            Window.currentRow - (window - 1), Window.currentRow
        )
        if method == "count":
            if isinstance(op_col, list):
                rolling = Rolling.__spark_methods[method]([c for c in op_col][0])
                return self.select(rolling)
            rolling = (
                Rolling.__spark_methods[method](op_col)
                .over(w)
                .alias(f"{op_col}_rolling_{method}")
            )
            return self.select("*", rolling)
        _unbounded_w = Window.orderBy(nat_order).rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        rolling = (
            when(
                row_number().over(_unbounded_w) >= min_p,
                Rolling.__spark_methods[method](op_col).over(w),
            )
            .otherwise(lit(None))
            .alias(f"{op_col}_rolling_{method}")
        )
        return self.select("*", rolling)


class RollingWrapper:
    """
    Class RollingWrapper is a class for wrapping methods from Rolling class
    adding this functionality to Spark DataFrame class
    """

    _wrapper(wrap_object=Rolling.rolling_down)


class DynamicSampling:

    __spark = OphelianSpark().ophelia_active_session()

    @staticmethod
    def empty_scan(self):
        cols = self.columns
        schema = StructType(
            [StructField(col_name, StringType(), True) for col_name in cols]
        )
        return self.sql_ctx.createDataFrame(self.sql_ctx._sc.emptyRDD(), schema)

    @staticmethod
    def __id_row_number(df, alias):
        return df.select("*", monotonically_increasing_id().alias(alias))

    @staticmethod
    def sample_n(self, n):
        _ = DynamicSampling.__id_row_number(self, "n")
        max_n = _.select("n").orderBy(col("n").desc()).limit(1).cache()
        sample_list = []
        for sample in range(n):
            # In this case collect() operation is permitted since we're collecting one single row
            sample_list.append(
                _.where(col("n") == random.randint(0, max_n.collect()[0][0]))
            )
        DynamicSampling.__spark.catalog.clearCache()
        return union_all(sample_list).drop("n")


class DynamicSamplingWrapper:
    """
    Class DynamicSamplingWrapper is a class for wrapping methods from DynamicSampling class
    adding this functionality to Spark DataFrame class
    """

    func = (DynamicSampling.empty_scan, DynamicSampling.sample_n)
    _wrapper(wrap_object=func)


class Selects:

    @staticmethod
    def regex_expr(regex_name):
        if isinstance(regex_name, list):
            return [f".*{regex}" for regex in regex_name]
        return [f".*{regex_name}"]

    @staticmethod
    def select_regex(self, reg_expr):
        compiled_regexes = [re.compile(regex) for regex in reg_expr]
        filtered_stream = filter(
            lambda line: any(regex.match(line) for regex in compiled_regexes),
            self.columns,
        )
        clean_regex_list = set(filtered_stream)
        return self.select(clean_regex_list)

    @staticmethod
    def select_startswith(self, regex):
        cols_list = self.columns
        if isinstance(regex, list):
            return self.select(
                [c for c in cols_list for reg in regex if c.startswith(reg)]
            )
        return self.select([c for c in cols_list if c.startswith(regex)])

    @staticmethod
    def select_endswith(self, regex):
        cols_list = self.columns
        if isinstance(regex, list):
            return self.select(
                [c for c in cols_list for reg in regex if c.endswith(reg)]
            )
        return self.select([c for c in cols_list if c.endswith(regex)])

    @staticmethod
    def select_contains(self, regex):
        if isinstance(regex, list):
            contain_cols = [c for c in self.columns for reg in regex if reg in c]
            clean_regex_list = remove_duplicate_element(contain_cols)
            return self.select(clean_regex_list)
        contain_cols = [c for c in self.columns if regex in c]
        clean_regex_list = remove_duplicate_element(contain_cols)
        return self.select(clean_regex_list)

    @staticmethod
    def sort_columns_asc(self):
        """
        DataFrame(self._jdf.select(self._jdf.schema.fields), self.sql_ctx)

        cols = self._jdf.column
        print("java jdf object", cols)
        print("java jdf type", cols.__class__.__name__)
        return DataFrame(self._jdf.select(self._sc, cols), self.sql_ctx)
        """
        return self.select(*sorted(self.columns))

    @staticmethod
    def select_freqitems(self, cols, support=None):
        if isinstance(cols, tuple):
            cols = list(cols)
        if not isinstance(cols, list):
            raise ValueError("cols must be a list or tuple of column names as strings.")
        if not support:
            support = 0.01
        return DataFrame(
            self._jdf.stat().freqItems(_to_seq(self._sc, cols), support), self.sql_ctx
        )

    @staticmethod
    def __type_list(datatype, select_type):
        return [k for k, v in datatype if v in select_type]

    @staticmethod
    def select_strings(self):
        dtype = self.dtypes
        stype = ["string", "str", "AnyStr", "char"]
        return Selects.__type_list(datatype=dtype, select_type=stype)

    @staticmethod
    def select_integers(self):
        dtype = self.dtypes
        stype = ["int", "integer"]
        return Selects.__type_list(datatype=dtype, select_type=stype)

    @staticmethod
    def select_floats(self):
        dtype = self.dtypes
        stype = ["float"]
        return Selects.__type_list(datatype=dtype, select_type=stype)

    @staticmethod
    def select_doubles(self):
        dtype = self.dtypes
        stype = ["double"]
        return Selects.__type_list(datatype=dtype, select_type=stype)

    @staticmethod
    def select_decimals(self):
        dtype = self.dtypes
        stype = ["decimal"]
        return Selects.__type_list(datatype=dtype, select_type=stype)

    @staticmethod
    def select_longs(self):
        dtype = self.dtypes
        stype = ["long", "bigint"]
        return Selects.__type_list(datatype=dtype, select_type=stype)

    @staticmethod
    def select_dates(self):
        dtype = self.dtypes
        stype = ["date", "timestamp"]
        return self.select(Selects.__type_list(datatype=dtype, select_type=stype))

    @staticmethod
    def select_complex(self):
        dtype = self.dtypes
        stype = ["complex"]
        return self.select(Selects.__type_list(datatype=dtype, select_type=stype))

    @staticmethod
    def select_structs(self):
        dtype = self.dtypes
        stype = ["list", "tuple", "array", "vector"]
        return self.select(Selects.__type_list(datatype=dtype, select_type=stype))

    @staticmethod
    def select_categorical(self):
        dtype = self.dtypes
        stype = ["string", "long", "bigint"]
        return self.select(Selects.__type_list(datatype=dtype, select_type=stype))

    @staticmethod
    def select_numerical(self):
        dtype = self.dtypes
        stype = ["double", "decimal", "integer", "int", "float", "complex"]
        return self.select(Selects.__type_list(datatype=dtype, select_type=stype))

    def select_features(self, df):
        return {
            "string": self.select_strings(df),
            "int": self.select_integers(df),
            "long": self.select_longs(df),
            "double": self.select_doubles(df),
            "float": self.select_floats(df),
            "date": self.select_dates(df),
            "complex": self.select_complex(df),
            "struct": self.select_structs(df),
            "categorical": self.select_categorical(df),
            "numeric": self.select_numerical(df),
        }


class SelectsWrapper:
    """
    Class SelectsWrapper is a class for wrapping methods from Selects class
    adding this functionality to Spark DataFrame class
    """

    func = (
        Selects.select_regex,
        Selects.select_startswith,
        Selects.select_endswith,
        Selects.select_contains,
        Selects.sort_columns_asc,
        Selects.select_freqitems,
        Selects.select_strings,
        Selects.select_integers,
        Selects.select_floats,
        Selects.select_doubles,
        Selects.select_decimals,
        Selects.select_longs,
        Selects.select_dates,
        Selects.select_features,
    )
    _wrapper(wrap_object=func)


class MapItem:

    @staticmethod
    def map_item(self, map_expr, origin_col):
        DataFrame.metadata = {"cols_to_drop": origin_col}
        if type(origin_col) is list:
            for c in origin_col:
                self = self.withColumn(c + "_bin", map_expr[self[c]])
        else:
            self = self.withColumn(origin_col + "_bin", map_expr[self[origin_col]])
        return self


class MapItemWrapper:
    """
    Class MapItemWrapper is a class for wrapping methods from MapItem class
    adding this functionality to Spark DataFrame class
    """

    _wrapper(wrap_object=MapItem.map_item)


class Reshape(DataFrame):

    __spark_methods = SparkMethods()

    def __init__(self, df):
        super().__init__(df._jdf, df.sql_ctx)
        self._df = df

    @staticmethod
    def narrow_format(self, fix_cols, new_cols=None):
        """
        Narrow format method will reshape from wide tabular table to
        narrow multidimensional format table for increasing push-down
        predicate native Spark performance
        :param self: Spark DataFrame inheritance object
        :param fix_cols: str, column name as fix column
        :param new_cols: str or list, with pivot and value new column names
        :return: narrow Spark DataFrame
        """
        try:
            if isinstance(new_cols, str):
                pivot_col, value_col = new_cols.split(",")[0], new_cols.split(",")[1]
            elif new_cols is not None and isinstance(new_cols, list):
                pivot_col, value_col = new_cols[0], new_cols[1]
            else:
                pivot_col, value_col = "no_name_pivot_col", "no_name_value_col"
            cols, dtype = zip(
                *[(c, t) for (c, t) in self.dtypes if c not in [fix_cols]]
            )

            generator_explode = explode(
                array(
                    [
                        struct(lit(c).alias(pivot_col), col(c).alias(value_col))
                        for c in cols
                    ]
                )
            ).alias("column_explode")
            column_to_explode = [
                f"column_explode.{pivot_col}",
                f"column_explode.{value_col}",
            ]

            return Reshape(
                self.select([fix_cols] + [generator_explode]).select(
                    [fix_cols] + column_to_explode
                )
            )
        except Exception as e:
            raise OphelianFunctionsException(
                f"An error occurred while calling narrow_format() method: {e}"
            )

    @staticmethod
    def narrow_format_v2(self, fix_cols, new_cols=None):
        try:
            if new_cols is None:
                pivot_col, value_col = "no_name_pivot_col", "no_name_value_col"
            elif isinstance(new_cols, str):
                pivot_col, value_col = new_cols.split(",")[0], new_cols.split(",")[1]
            elif isinstance(new_cols, list):
                pivot_col, value_col = new_cols[0], new_cols[1]
            cols, dtype = zip(
                *[(c, t) for (c, t) in self.dtypes if c not in [fix_cols]]
            )
            generator_explode = explode(
                array(
                    [
                        struct(lit(c).alias(pivot_col), col(c).alias(value_col))
                        for c in cols
                    ]
                )
            ).alias("column_explode")
            column_to_explode = (
                f"column_explode.{pivot_col}",
                f"column_explode.{value_col}",
            )
            return self.selectExpr(fix_cols, *column_to_explode)
        except Exception as e:
            raise OphelianFunctionsException(
                f"An error occurred while calling narrow_format() method: {e}"
            )

    @staticmethod
    def wide_format(
        self, group_by, pivot_col, agg_dict, rnd=6, rep=20, rename_col=False
    ):
        """
        Wide format method will reshape from narrow multidimensional
        table to wide tabular format table for feature wide table
        :param self: Spark DataFrame inheritance object
        :param group_by: str, name for group DataFrame
        :param pivot_col: str, name for wide pivot column
        :param agg_dict: dict, with wide Spark function
        :param rnd: int, for round wide value
        :param rep: int, repartition threshold for wide partition optimization
        :param rename_col: bool, boolean response for rename existing columns
        :return: wide Spark DataFrame
        """
        try:
            agg_list = []
            keys, values = list(agg_dict.keys())[0], list(agg_dict.values())[0]

            for k in agg_dict:
                for i in range(len(agg_dict[k].split(","))):
                    strip_string = agg_dict[k].replace(" ", "").split(",")
                    agg_item = spark_round(
                        Reshape.__spark_methods[strip_string[i]](k), rnd
                    ).alias(f"{strip_string[i]}_{k}")
                    agg_list.append(agg_item)

            if isinstance(group_by, list):
                if rename_col:
                    group_by_expr = [col(c).alias(f"{c}_{pivot_col}") for c in group_by]
                else:
                    group_by_expr = [col(c).alias(f"{c}") for c in group_by]
            else:
                if rename_col:
                    group_by_expr = col(group_by).alias(f"{group_by}_{pivot_col}")
                else:
                    group_by_expr = col(group_by).alias(f"{group_by}")

            if len(list(agg_dict)) == 1:
                pivot_df = (
                    self.groupBy(group_by_expr)
                    .pivot(pivot_col)
                    .agg(*agg_list)
                    .na.fill(0)
                )

                if rename_col:
                    renamed_cols = [
                        col(c).alias(f"{c}_{keys}_{values}")
                        for c in pivot_df.columns[1:]
                    ]
                    return Reshape(
                        pivot_df.select(f"{group_by}_{pivot_col}", *renamed_cols)
                    )
                else:
                    renamed_cols = [col(c).alias(f"{c}") for c in pivot_df.columns[1:]]
                    return Reshape(pivot_df.select(f"{group_by}", *renamed_cols))
            return Reshape(
                self.groupBy(group_by_expr).pivot(pivot_col).agg(*agg_list).na.fill(0)
            )
        except TypeError as te:
            raise OphelianFunctionsException(
                f"An error occurred while calling wide_format() method: {te}"
            )


class ReshapeWrapper:
    """
    Class ReshapeWrapper is a class for wrapping methods from Reshape class
    adding this functionality to Spark DataFrame class
    """

    func = (Reshape.wide_format, Reshape.narrow_format)
    _wrapper(wrap_object=func)


class PctChange:

    @staticmethod
    def __build_pct_change_function(x, t, w):
        if isinstance(x, str):
            list_x = [x]
            return list(
                map(
                    lambda c: (col(c) / lag(col(c), offset=t).over(w) - 1).alias(c),
                    list_x,
                )
            )
        elif isinstance(x, list):
            return list(
                map(lambda c: (col(c) / lag(col(c), offset=t).over(w) - 1).alias(c), x)
            )

    @staticmethod
    def __build_window_partition(
        self, periods, order_by, pct_cols, partition_by=None, desc=False
    ):
        if desc:
            if partition_by is None:
                win = Window.orderBy(col(order_by).desc())
            else:
                win = Window.partitionBy(partition_by).orderBy(col(order_by).desc())
        else:
            if partition_by is None:
                win = Window.orderBy(col(order_by))
            else:
                win = Window.partitionBy(partition_by).orderBy(col(order_by))
        function = PctChange.__build_pct_change_function(x=pct_cols, t=periods, w=win)
        return self.select(function)

    @staticmethod
    def __infer_sort_column(self):
        regex_match = ["year", "Year", "date", "Date", "month", "Month", "day", "Day"]
        regex_list = self.selectRegex(regex_expr(regex_match)).columns
        f_pick = feature_pick(self)
        if len(regex_list) > 0:
            return regex_list
        elif len(f_pick["date"]) > 0:
            return f_pick["date"]
        else:
            return f_pick["string"]

    @staticmethod
    def __infer_lag_column(self):
        f_pick = feature_pick(self)
        if len(f_pick["double"]) > 0:
            return f_pick["double"]
        elif len(f_pick["float"]) > 0:
            return f_pick["float"]
        elif len(f_pick["long"]) > 0:
            return f_pick["long"]
        else:
            return f_pick["int"]

    @staticmethod
    def __infer_lag_columnv2(df):
        numeric_cols = df.select(
            lambda col_name: df[col_name].dtype in (float, int)
        ).columns
        if numeric_cols:
            return numeric_cols[0]
        else:
            return None

    @staticmethod
    def pct_change(
        self, periods=1, order_by=None, pct_cols=None, partition_by=None, desc=None
    ):
        if (order_by is None) and (pct_cols is None):
            infer_sort = PctChange.__infer_sort_column(self)[0]
            infer_lag = PctChange.__infer_lag_column(self)[1]
            return PctChange.__build_window_partition(
                self, periods, infer_sort, infer_lag
            )
        return PctChange.__build_window_partition(
            self, periods, order_by, pct_cols, partition_by, desc
        )

    @staticmethod
    def remove_element(self, col_remove):
        primary_list = self.columns
        [primary_list.remove(c) for c in col_remove]
        return primary_list


class PctChangeWrapper:
    """
    Class PctChangeWrapper is a class for wrapping methods from PctChange class
    adding this functionality to Spark DataFrame class
    """

    func = (PctChange.pct_change, PctChange.remove_element)
    _wrapper(wrap_object=func)


class CrossTabular:

    @staticmethod
    def __expression(cols_list, xpr):
        expr_dict = {
            "sum": "+".join(cols_list),
            "sub": "-".join(cols_list),
            "mul": "*".join(cols_list),
            "div": "/".join(cols_list),
        }
        return expr_dict[xpr]

    @staticmethod
    def foreach_col(self, group_by, pivot_col, agg_dict, oper):
        func = []
        regex_keys = list(agg_dict.keys())
        regex_values = list(agg_dict.values())
        df = self.toWide(group_by, pivot_col, agg_dict)
        for i in range(len(regex_keys)):
            cols_list = df.selectRegex(regex_expr(regex_keys[i])).columns
            expression = expr(CrossTabular.__expression(cols_list, oper))
            func.append(expression.alias(f"{regex_keys[i]}_{regex_values[i]}_{oper}"))
        return df.select("*", *func)

    @staticmethod
    def foreach_colv2(self, group_by, pivot_col, agg_dict, oper):
        df = self.toWide(group_by, pivot_col, agg_dict)
        df = df.groupBy(group_by).pivot(pivot_col).agg(agg_dict)
        df = df.groupBy(group_by).agg(expr(f"{oper}({', '.join(agg_dict.keys())})"))
        df = df.select(
            concat_ws("_", *[col(c).alias(c) for c in df.columns]).alias("new_col")
        )
        return df

    @staticmethod
    def resume_dataframe(self, group_by=None, new_col=None):
        cols_types = [k for k, v in self.dtypes if v != "string"]
        if group_by is None:
            try:
                agg_df = self.agg(*[sum(c).alias(c) for c in cols_types])
                return agg_df.withColumn(new_col, lit("+++ total")).select(
                    new_col, *cols_types
                )
            except Py4JJavaError as e:
                raise AssertionError(f"empty expression found. {e}")
        return self.groupBy(group_by).agg(*[sum(c).alias(c) for c in cols_types])

    @staticmethod
    def resume_dataframev2(self, group_by=None, new_col=None):

        if self.rdd.isEmpty():
            return self.sql_ctx.createDataFrame([], self.schema)

        cols_types = [k for k, v in self.dtypes if v != "string"]

        if group_by is not None and group_by in self.columns:
            if self.columns.count(group_by) > 1:
                raise AssertionError(f"duplicate column found: {group_by}")
            if self.groupBy(group_by)._jdf.isStreaming():
                raise AssertionError(
                    f"the dataframe is not grouped by column: {group_by}"
                )

        if group_by is None:
            try:
                agg_df = self.agg(*[sum(c).alias(c) for c in cols_types])
                return agg_df.withColumn(new_col, lit("+++ total")).select(
                    new_col, *cols_types
                )
            except Py4JJavaError as e:
                raise AssertionError(f"empty expression found. {e}")
        return self.groupBy(group_by).agg(*[sum(c).alias(c) for c in cols_types])

    @staticmethod
    def tab_table(self, group_by, pivot_col, agg_dict, oper="sum"):
        sum_by_col_df = CrossTabular.foreach_col(
            self, group_by, pivot_col, agg_dict, oper
        )
        return sum_by_col_df.union(
            CrossTabular.resume_dataframe(sum_by_col_df, new_col=self.columns[0])
        )

    @staticmethod
    def cross_pct(self, group_by, pivot_col, agg_dict, operand="sum", cols=None):
        sum_by_col_df = CrossTabular.foreach_col(
            self, group_by, pivot_col, agg_dict, operand
        )
        union_df = sum_by_col_df.union(
            CrossTabular.resume_dataframe(sum_by_col_df, new_col=self.columns[0])
        )
        key_list = list(agg_dict.keys())
        func = []
        for i in range(len(key_list)):
            tmp_df = sum_by_col_df.selectRegex(regex_expr([key_list[i]]))
            fix_df = tmp_df.selectRegex(regex_expr([operand]))
            pivot_list = tmp_df.drop(fix_df.columns[0]).columns
            for ix in range(len(pivot_list)):
                operate_cols = [pivot_list[ix], fix_df.columns[0]]
                dynamic_expr = CrossTabular.__expression(operate_cols, "div")
                func.append(
                    spark_round(expr(dynamic_expr), 4).alias(f"{pivot_list[ix]}_prop")
                )
        if cols == "all":
            return union_df.select("*", *func)
        else:
            return union_df.select(f"{group_by}_{pivot_col}", *func)

    @staticmethod
    def cross_pctv2(self, group_by, pivot_col, agg_dict, operand="sum", cols=None):
        grouped_df = self.groupBy(group_by, pivot_col).agg(agg_dict)
        grouped_df = grouped_df.withColumn(
            "proportion", grouped_df[operand] / grouped_df[operand].sum()
        )
        pivot_col_list = grouped_df.drop(grouped_df.columns[0:2]).columns
        for ix in range(len(pivot_col_list)):
            grouped_df = grouped_df.withColumnRenamed(
                pivot_col_list[ix], f"{pivot_col_list[ix]}_prop"
            )
        union_df = grouped_df.union(self)
        if cols == "all":
            return union_df
        else:
            return union_df.select(f"{group_by}_{pivot_col}", *pivot_col_list)


class CrossTabularWrapper:
    """
    Class CrossTabularWrapper is a class for wrapping methods from CrossTabular class
    adding this functionality to Spark DataFrame class
    """

    func = (
        CrossTabular.foreach_col,
        CrossTabular.resume_dataframe,
        CrossTabular.tab_table,
        CrossTabular.cross_pct,
    )
    _wrapper(wrap_object=func)


class Joins:

    @staticmethod
    def join_small_right(self, small_df, on, how):
        """
        Join Small Right wrapper broadcasts small sized DataFrames generating a copy of the same
        DataFrame in every worker.
        :param self: heritage DataFrame class object
        :param small_df: refers to the small size DataFrame to copy
        :param on: str for the join column name or a list of Columns
        :param how: str default 'inner'. Must be one of: {'inner', 'cross', 'outer',
        'full', 'fullouter', 'full_outer', 'left', 'leftouter', 'left_outer',
        'right', 'rightouter', 'right_outer', 'semi', 'leftsemi', 'left_semi',
        'anti', 'leftanti', 'left_anti'}
        :return

        Example:
        ' >>> big_df.join_small_right(small_df, on='On_Var', how='left') '
        """
        return self.join(broadcast(small_df), on, how)

    @staticmethod
    def join_small_left(self, df, on, how):
        """
        Join Small Left wrapper broadcasts small sized DataFrames generating a copy of the same
        DataFrame in every worker.
        :param self: heritage DataFrame class object
        :param df: refers to the big size DataFrame to copy
        :param on: str for the join column name or a list of Columns
        :param how: str default 'inner'. Must be one of: {'inner', 'cross', 'outer',
        'full', 'fullouter', 'full_outer', 'left', 'leftouter', 'left_outer',
        'right', 'rightouter', 'right_outer', 'semi', 'leftsemi', 'left_semi',
        'anti', 'leftanti', 'left_anti'}
        :return

        Example:
        ' >>> small_df.join_small_left(big_df, on='On_Var', how='left') '
        """
        return df.join(broadcast(self), on, how)


class JoinsWrapper:
    """
    Class JoinsWrapper is a class for wrapping methods from Joins class
    adding this functionality to Spark DataFrame class
    """

    func = (Joins.join_small_right, Joins.join_small_left)
    _wrapper(wrap_object=func)


class DaskSpark:

    @staticmethod
    def __file_system(df):
        sql_ctx = df.sql_ctx
        _sc = sql_ctx and sql_ctx._sc
        fl = _sc._jvm.org.apache.hadoop.fs.FileSystem
        fs = fl.get(_sc._jsc.hadoopConfiguration())
        path = _sc._jvm.org.apache.hadoop.fs.Path
        return {"file_sys": fl, "hadoop_fs": fs, "fs_path": path}

    @staticmethod
    def __fs_clean(self, clean_path):
        # File System instance
        env_fs = DaskSpark.__file_system(self)
        # HDFS command to delete file paths
        env_fs["hadoop_fs"].delete(env_fs["fs_path"](clean_path))

    @staticmethod
    def __fs_rename(self, path, rename):
        # File System instance
        env_fs = DaskSpark.__file_system(self)
        # HDFS command to delete file paths
        env_fs["hadoop_fs"].rename(env_fs["fs_path"](path), env_fs["fs_path"](rename))

    @staticmethod
    def dask_read(option, file_path):

        # Python map for file type pattern
        file_type = {
            "parquet": file_path + "/*.parquet",
            "csv": file_path + "/*.csv",
            "json": file_path + "/*.json",
            "text": file_path + "/*.txt",
        }

        # Define reader type by pattern mapping
        file_pattern = file_type[option]
        dask_reader = {
            "parquet": dask_df.read_parquet(file_pattern, engine="pyarrow"),
            "csv": dask_df.read_csv(file_pattern),
            "json": dask_df.read_json(file_pattern),
            "text": dask_df.read_table(file_pattern),
        }

        return dask_reader[option]

    @staticmethod
    def spark_to_dask(self, option="csv", mode="overwrite", checkpoint_path=None):
        try:
            sc = SparkContext._active_spark_context
            spark = SparkSession(sc)
        except Exception:
            sc = SparkContext.getOrCreate()
            spark = SparkSession(sc)

        # Write Spark DataFrame to parquet
        work_dir = os.getcwd() + "/data/stream/dask"
        tmp_dir = work_dir + "/tmp"

        # Lets leave 'overwrite' config as default config
        self.write.mode("overwrite").parquet(tmp_dir)

        # Retrieve DataFrame schema
        schema_parquet = spark.read.parquet(tmp_dir).schema

        if checkpoint_path is None:
            checkpoint_path = tmp_dir + "_checkpoint_data"

        stream_option = {
            "text": spark.readStream.schema(schema_parquet).text(tmp_dir),
            "csv": spark.readStream.schema(schema_parquet).csv(tmp_dir),
            "json": spark.readStream.schema(schema_parquet).json(tmp_dir),
            "parquet": spark.readStream.schema(schema_parquet).parquet(tmp_dir),
        }

        stream_writer = (
            stream_option[option]
            .coalesce(1)
            .writeStream.format(option)
            .outputMode("append")
            .queryName("stream_query")
            .option("checkpointLocation", checkpoint_path)
        )

        file_path = work_dir + "/tmp_file"
        stream_query = stream_writer.start(file_path)
        stream_query.processAllAvailable()

        if mode == "overwrite":
            DaskSpark.__fs_clean(self, tmp_dir)
            DaskSpark.__fs_clean(self, checkpoint_path)
            DaskSpark.__fs_rename(self, file_path, work_dir + "/file")
            DaskSpark.__fs_clean(self, file_path)

        return DaskSpark.dask_read(option, file_path)

    @staticmethod
    def spark_to_series(self, column_series):
        dask_df = DaskSpark.spark_to_dask(self)
        series = dask_df[column_series]
        list_dask = series.to_delayed()
        full = [
            dask_arr.from_delayed(i, i.compute().shape, i.compute().dtype)
            for i in list_dask
        ]
        return dask_arr.concatenate(full)

    @staticmethod
    def spark_to_numpy(self, column_series=None):
        if column_series is not None:
            dask_array = self.toPandasSeries(column_series)
            return dask_arr.from_array(dask_array.compute())
        else:
            dask_pandas_series = DaskSpark.spark_to_dask(self)
            return dask_pandas_series.to_dask_array()


class DaskSparkWrapper:
    """
    Class DaskSparkWrapper is a class for wrapping methods from DaskSpark class
    adding this functionality to Spark DataFrame class
    """

    func = (
        DaskSpark.spark_to_dask,
        DaskSpark.spark_to_series,
        DaskSpark.spark_to_numpy,
    )
    _wrapper(wrap_object=func)


class SortinoRatioCalculator:
    def __init__(self, df: DataFrame):
        self.df = df

    def return_on_investment(self):
        df = self.df.withColumn("prev_close", lag("close").over(Window.orderBy("date")))
        return df.withColumn("roi", (df["close"] - df["prev_close"]) / df["prev_close"])

    def downside_deviation(self, df):
        negative_roi = df.filter(df["roi"] < 0).select("roi")
        downside_deviation = negative_roi.select(stddev("roi")).first()[0]
        return downside_deviation

    def sortino_ratio(self):
        df = self.return_on_investment()
        downside_deviation = self.downside_deviation(df)
        positive_roi = df.filter(df["roi"] >= 0).select("roi")
        mean_positive_roi = positive_roi.select(mean("roi")).first()[0]
        sortino_ratio = mean_positive_roi / downside_deviation
        return sortino_ratio


class SortinoRatioCalculatorWrapper:
    """
    Class SortinoRatioCalculatorWrapper is a class for wrapping methods from SortinoRatioCalculator class
    adding this functionality to Spark DataFrame class.
    """

    func = SortinoRatioCalculator.sortino_ratio
    _wrapper(wrap_object=func)


class SharpeRatioCalculator:
    def __init__(self, data: DataFrame, returns_col: str, risk_free_rate: float):
        self.data = data
        self.returns_col = returns_col
        self.risk_free_rate = risk_free_rate

    def calculate(self):
        """
        Calculate the Sharpe ratio for the given data.
        """
        mean_return = self.data.select(mean(self.returns_col)).first()[0]
        std_dev = self.data.select(stddev(self.returns_col)).first()[0]
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_dev
        return sharpe_ratio


class SharpeRatioCalculatorWrapper:
    """
    Class SharpeRatioCalculatorWrapper is a class for wrapping methods from SharpeRatioCalculator class
    adding this functionality to Spark DataFrame class.
    """

    func = SharpeRatioCalculator.calculate
    _wrapper(wrap_object=func)


class EfficientFrontierRatioCalculator:
    def __init__(self, dataframe):
        self.df = dataframe.cache()

    def expected_returns(self):
        returns = self.df.agg(mean("return")).first()[0]
        return returns

    def expected_variances(self):
        cov_matrix = self.df.agg(variance("return")).first()[0]
        return cov_matrix

    @staticmethod
    def efficient_frontier(cov_matrix, returns):
        weights = solve_qp(cov_matrix, -returns, None, None)
        return weights

    def calculate_efficient_frontier_ratio(self):
        returns = self.expected_returns()
        cov_matrix = self.expected_variances()
        weights = self.efficient_frontier(cov_matrix, returns)

        min_var_ret = weights @ returns
        max_ret_var = weights @ cov_matrix @ weights
        eff_frontier_ratio = min_var_ret / max_ret_var
        return eff_frontier_ratio


class EfficientFrontierRatioCalculatorWrapper:
    """
    Class EfficientFrontierRatioCalculatorWrapper is a class for wrapping
    methods from EfficientFrontierRatioCalculator class
    adding this functionality to Spark DataFrame class.
    """

    func = EfficientFrontierRatioCalculator.calculate_efficient_frontier_ratio
    _wrapper(wrap_object=func)


class RiskParityCalculator:
    def __init__(
        self,
        returns_columns,
        asset_column,
        weight_columns,
        risk_columns,
        dataframe: DataFrame,
        sql_context: SQLContext,
    ):
        self.returns_columns = returns_columns
        self.asset_column = asset_column
        self.weight_columns = weight_columns
        self.risk_columns = risk_columns
        self.dataframe = dataframe
        self.sql_context = sql_context

    def calculate_asset_risk(self):
        returns_df = self.dataframe.select(self.returns_columns)
        std_df = returns_df.agg(
            *[stddev(c).alias(c + "_std") for c in returns_df.columns]
        )
        var_df = returns_df.agg(
            *[variance(c).alias(c + "_var") for c in returns_df.columns]
        )
        return self.dataframe.join(std_df, on=self.asset_column).join(
            var_df, on=self.asset_column
        )

    def calculate_total_risk(self):
        risk_df = self.dataframe.select(self.weight_columns + self.risk_columns)
        for col in self.risk_columns:
            risk_df = risk_df.withColumn(
                col + "_weighted", col * risk_df[self.weight_columns]
            )
        total_risk_df = risk_df.agg(sum(c(c + "_weighted") for c in self.risk_columns))
        return total_risk_df

    def calculate_risk_parity_weights(self):
        # Define the objective function and the constraints
        def obj_func(w):
            return sum(
                self.dataframe[c + "_weighted"].dot(w) for c in self.risk_columns
            )

        def grad_obj_func(ws):
            return np.array(
                [self.dataframe[c + "_weighted"].dot(ws) for c in self.risk_columns]
            )

        def constraint_func(we):
            return sum(
                when(self.dataframe[c] == 1, 1.0).otherwise(0.0).dot(we)
                for c in self.weight_columns
            )

        def grad_constraint_func(wes):
            return np.array(
                [
                    when(self.dataframe[c] == 1, 1.0).otherwise(0.0).dot(wes)
                    for c in self.weight_columns
                ]
            )

        # Initialize the LBFGS optimization
        optimizer = LBFGS(
            x0=np.zeros(len(self.weight_columns)),
            func=obj_func,
            grad=grad_obj_func,
            m=5,
        )

        # Minimize the objective function subject to the constraints
        weights = optimizer.minimize(
            constraint_func=constraint_func, grad_constraint_func=grad_constraint_func
        )

        # Return the optimized weights
        return weights

    def update_weights(self, risk_parity_weights_df):
        updated_df = self.dataframe.join(risk_parity_weights_df, on=self.asset_column)
        updated_df = (
            updated_df.withColumn(
                self.weight_columns, updated_df[self.weight_columns + "_risk_parity"]
            )
            .drop(self.weight_columns + "_risk_parity")
            .withColumnRenamed(self.weight_columns, self.weight_columns + "_original")
        )
        return updated_df


class RiskParityCalculatorWrapper:
    """
    Class RiskParityCalculatorWrapper is a class for wrapping
    methods from RiskParityCalculator class
    adding this functionality to Spark DataFrame class.
    """

    func = RiskParityCalculator.calculate_risk_parity_weights
    _wrapper(wrap_object=func)


if __name__ == "__main__":
    # Create a sample DataFrame using PySpark
    spark = SparkSession.builder.appName("example").getOrCreate()

    data = [("A", 1), ("B", None), ("C", 3), ("A", None), ("B", 5), ("C", 6)]
    columns = ["partition", "values"]
    sdf = spark.createDataFrame(data, columns)

    # Clean null values
    clean_df = NullDebug.null_clean(sdf, "partition", 0.5)
    clean_df.show()
