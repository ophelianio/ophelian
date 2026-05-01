from functools import lru_cache, reduce
from typing import List

import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import (
    asc,
    col,
    dayofmonth,
    desc,
    month,
    row_number,
    udf,
    year,
)
from pyspark.sql.types import (
    IntegerType,
    LongType,
    Row,
    StructField,
    StructType,
    TimestampType,
)

from ophelian._logger import OphelianLogger
from ophelian.ophelian_spark import OphelianUtilitiesException

__all__ = [
    "union_all",
    "split_date",
    "row_index",
    "lag_min_max_data",
    "regex_expr",
    "remove_duplicate_element",
    "year_array",
    "dates_index",
    "sorted_date_list",
    "feature_pick",
    "binary_search",
    "century_from_year",
    "simple_average",
    "delta_series",
    "simple_moving_average",
    "average",
    "weight_moving_average",
    "single_exp_smooth",
    "double_exp_smooth",
    "initial_seasonal_components",
    "triple_exp_smooth",
    "row_indexing",
    "string_match",
]
logger = OphelianLogger()


def union_all(dfs: list):
    """
    Union all helps to gather in one single spark DataFrame a list of multiple DF
    :param dfs: list of dataframes
    :return: union df
    """
    try:
        first = dfs[0]
        union_dfs = first.sql_ctx._sc.union([df.cache().rdd for df in dfs])
        return first.sql_ctx.createDataFrame(union_dfs, first.schema)
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on union_all() method: {e}"
        )


def split_date(df, col_date: str):
    """
    Split date helps to divide date field into day, month and year by each column
    :param df: spark DataFrame with date field to split
    :param col_date: str, column name date
    :return: spark DataFrame
    """
    try:
        dates_df = df.select(
            "*",
            year(col_date).alias(f"{col_date}_year"),
            month(col_date).alias(f"{col_date}_month"),
            dayofmonth(col_date).alias(f"{col_date}_day"),
        )
        logger.info("Split Date In Columns")
        return dates_df
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on split_date() method: {e}"
        )


def row_index(df, col_order: str):
    """
    Row index method will help to create a row index for a given spark DataFrame
    :param df: data to analyze
    :param col_order: column to order
    :return: DataFrame
    """
    try:
        w = Window().orderBy(col(col_order).desc())
        logger.info("Row Indexing In DataFrame")
        return df.withColumn("row_num", row_number().over(w))
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on row_index() method: {e}"
        )


def lag_min_max_data(df, is_max=True, col_lag: str = "operation_date"):
    """
    This is a placeholder for this method
    :param df: data to analyze
    :param is_max: indicates if it is max
    :param col_lag: name of the column to lag
    :return: DataFrame
    """
    try:
        if is_max:
            lag_date = df.agg(max(col(col_lag).cast(TimestampType()))).first()[0]
        else:
            lag_date = df.agg(min(col(col_lag).cast(TimestampType()))).first()[0]
        lag_data = df.where(col(col_lag) < lag_date).select(
            [col(c).alias(f"{c}_lag") for c in df.columns]
        )
        logger.info("Lag-Over Dates In DataFrame")
        return lag_data
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on lag_min_max_data() method: {e}"
        )


def regex_expr(regex_name):
    """
    Creates a regex expression for one or more regex
    :param regex_name: str or list(str) regex character to find
    :return: list
    """
    try:
        if isinstance(regex_name, list):
            return [f".*{re}" for re in regex_name]
        return [f".*{regex_name}"]
    except ValueError as ve:
        raise OphelianUtilitiesException(
            f"An error occurred on regex_expr() method: {ve}"
        )


def remove_duplicate_element(lst: list):
    """
    Remove duplicate element in given array
    :param lst: list of n elements with duplicates
    :return: list
    """
    try:
        return list(dict.fromkeys(lst))
    except ValueError as ve:
        raise OphelianUtilitiesException(
            f"An error occurred on remove_duplicate_element() method: {ve}"
        )


def year_array(from_year, to_year):
    """
    Gets every year number between a range, including the upper limit
    :param from_year: start year number
    :param to_year: end year number
    :return: list
    """
    try:
        logger.info(f"Window Data From Year {from_year} To {to_year}")
        return list(range(int(from_year), int(to_year) + 1))
    except ValueError as ve:
        raise OphelianUtilitiesException(
            f"An error occurred on year_array() method: {ve}"
        )


def dates_index(dates_list: list):
    """
    Dates parser function, transform a list of dates in a dictionary
    :param dates_list: sequence of date values
    :return: callable function
    """
    try:
        if len(dates_list) == 0:
            raise AssertionError("Empty Parameters Array")
        dates_dict = {date: index for index, date in enumerate(dates_list)}
        logger.info("Set Date Index")
        return udf(lambda x: dates_dict[x], IntegerType())
    except ValueError as ve:
        raise OphelianUtilitiesException(
            f"An error occurred on dates_index() method: {ve}"
        )


@lru_cache(maxsize=60)
def sorted_date_list(df, col_collect: str):
    """
    Builds a sorted list of every value for a date column in a given DataFrame
    :param df: data to analyze
    :param col_collect: column to analyze
    :return: list
    """
    try:
        logger.info("Order Date List")
        return sorted(
            [x.operation_date for x in df.select(col_collect).distinct().collect()]
        )
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on sorted_date_list() method: {e}"
        )


@lru_cache(maxsize=60)
def sorted_date_listv2(df, col_collect: str):
    """
    Builds a sorted list of every value for a date column in a given DataFrame
    :param df: data to analyze
    :param col_collect: column to analyze
    :return: list
    """
    try:
        logger.info("Order Date List")
        unique_date = set(df[col_collect])
        unique_date = list(unique_date)
        unique_date.sort()
        return unique_date
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on sorted_date_listv2() method: {e}"
        )


def feature_pick(df):
    """
    Feature pick function helps to split variable names from spark DataFrame
    into 'string', 'int', 'bigint', 'double', 'float', 'date' and 'other' type in separated list
    :param df: spark DataFrame with fields to analyze
    :return: dict
    """
    try:
        s, i, l, d, f, t, o = [], [], [], [], [], [], []
        for k, v in df.dtypes:
            (
                s.append(k)
                if v in ["str", "string"]
                else (
                    i.append(k)
                    if v in ["int", "integer"]
                    else (
                        l.append(k)
                        if v in ["bigint", "long"]
                        else (
                            d.append(k)
                            if v in ["double"]
                            else (
                                f.append(k)
                                if v in ["float"]
                                else (
                                    t.append(k)
                                    if v in ["date", "timestamp"]
                                    else o.append(k)
                                )
                            )
                        )
                    )
                )
            )
        return {
            "string": s,
            "int": i,
            "long": l,
            "double": d,
            "float": f,
            "date": t,
            "other": o,
        }
    except ValueError as ve:
        raise OphelianUtilitiesException(
            f"An error occurred on feature_pick() method: {ve}"
        )


def feature_pickv2(df):
    """
    Feature pick function helps to split variable names from spark DataFrame
    into 'string', 'int', 'bigint', 'double', 'float', 'date' and 'other' type in separated list
    :param df: spark DataFrame with fields to analyze
    :return: dict
    """
    try:
        column_types = {
            "string": [],
            "int": [],
            "long": [],
            "double": [],
            "float": [],
            "date": [],
            "other": [],
        }
        for k, v in df.dtypes:
            if v in ["str", "string"]:
                column_types["string"].append(k)
            elif v in ["int", "integer"]:
                column_types["int"].append(k)
            elif v in ["bigint", "long"]:
                column_types["long"].append(k)
            elif v in ["double"]:
                column_types["double"].append(k)
            elif v in ["float"]:
                column_types["float"].append(k)
            elif v in ["date", "timestamp"]:
                column_types["date"].append(k)
            else:
                column_types["other"].append(k)
        return column_types
    except ValueError as ve:
        raise OphelianUtilitiesException(
            f"An error occurred on feature_pickv2() method: {ve}"
        )


def __binary_helper_search(array, target, left_p, right_p):
    try:
        if left_p > right_p:
            raise AssertionError("None binary pointer")

        mid_point = (left_p + right_p) // 2
        potential_match = array[mid_point]

        if target == potential_match:
            return mid_point
        elif target < potential_match:
            return __binary_helper_search(array, target, left_p, mid_point - 1)
        else:
            return __binary_helper_search(array, target, mid_point + 1, right_p)
    except ValueError as ve:
        raise OphelianUtilitiesException(
            f"An error occurred on __binary_helper_search() private method: {ve}"
        )


def binary_search(array: list, target):
    """
    Use a helper recursive binary search method for O(n*log(n)) search items
    :param array: array of elements
    :param target: number to search
    :return: array index int
    """
    logger.info("Binary Find")
    return __binary_helper_search(array, target, 0, len(array) - 1)


def century_from_year(yr: int):
    """
    Calculates the century from a given year
    :param yr: int representing year
    :return: century int
    """
    try:
        return (yr - 1) // 100 + 1
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on century_from_year() method: {ae}"
        )


@lru_cache(maxsize=30)
def simple_average(series: List[float]):
    """
    Compute the simple average from a given series
    :param series: list of float observation series
    :return: float
    """
    try:
        logger.info("Compute Simple Average")
        return reduce(lambda a, b: a + b, series) / len(series)
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on simple_average() method: {ae}"
        )


def delta_series(series: List[float]):
    """
    Identify the delta variation from a given series
    :param series: list of float observation series
    :return: float
    """
    try:
        y, y_n = np.array(series), len(series)
        y_hat = simple_average(series)
        return float(
            2.048 * np.sqrt((1 / (y_n - 2)) * (sum((y - y_hat) ** 2) / np.var(y)))
        )
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on delta_series() method: {ae}"
        )


@lru_cache(maxsize=30)
def simple_moving_average(series: List[float], n_moving_day: int):
    """
    Compute the simple moving average (SMA) from a given series
    :param series: array of float observation series
    :param n_moving_day: int of n moving observations
    :return: float
    """
    try:
        logger.info("SMA")
        return simple_average(series=series[-n_moving_day:])
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on simple_moving_average() method: {ae}"
        )


def average(series: List[float], n_moving_day=None):
    """
    Wrapper for both average function type, simple average and SMA
    :param series: array of float observation series
    :param n_moving_day: int of n moving observations
    :return: float
    """
    try:
        if n_moving_day is None:
            return simple_average(series=series)
        return simple_moving_average(series=series, n_moving_day=n_moving_day)
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(f"An error occurred on average() method: {ae}")


@lru_cache(maxsize=30)
def weight_moving_average(series: List[float], weights: List[float]):
    """
    Compute weight moving average (WMA) from a given series
    :param series: array of float observation series
    :param weights: list of weights that must add up to 1, e.g. [0.1,0.2,0.3,0.4] = 1
    :return: float
    """
    try:
        if sum(weights) != 1:
            raise AssertionError("Invalid list, sum of weights must be equal to 1")
        result = 0.0
        weights.reverse()
        for n in range(len(weights)):
            result += series[-n - 1] * weights[n]
        return result
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on weight_moving_average() method: {ae}"
        )


@lru_cache(maxsize=30)
def single_exp_smooth(series: List[float], alpha: float = 0.05):
    """
    Compute single exponential smooth series with alpha data smooth coefficient
    :param series: array of float observation series
    :param alpha: float alpha smooth 0.05 set as default, other options could be: 0.5, 0.005, 0.0005
    :return: Python dict with series decomposition components
    """
    try:
        result = [series[0]]
        for n in range(1, len(series)):
            result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
        return {"single_exp_smooth": result, "delta": delta_series(result)}
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on single_exp_smooth() method: {ae}"
        )


@lru_cache(maxsize=30)
def double_exp_smooth(series: List[float], alpha: float = 0.05, beta: float = 0.005):
    """
    Compute double exponential smooth series with alpha data smooth and beta trend smooth coefficients
    :param series: array of float observation series
    :param alpha: float alpha data smooth factor 0.05 set as default, other options: 0.5, 0.005, 0.0005
    :param beta: float beta trend smooth factor 0.005 set as default
    :return: Python dict with series decomposition components
    """
    try:
        result = [series[0]]
        level, trend = series[0], (series[1] - series[0])
        for n in range(1, len(series) + 1):
            if n >= len(series):
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, (alpha * value + (1 - alpha) * (level + trend))
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        return {
            "double_exp_smooth": result,
            "level": level,
            "trend": trend,
            "delta": delta_series(result),
        }
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on double_exp_smooth() method: {ae}"
        )


@lru_cache(maxsize=30)
def __initial_trend(series, series_len):
    try:
        init_trend = 0.0
        for i in range(series_len):
            init_trend += float(series[i + series_len] - series[i]) / series_len
        return init_trend / series_len
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on __initial_trend() private method: {ae}"
        )


def initial_seasonal_components(series: List[float], slen: int = 15):
    """
    Compute the initial seasonal components for a given series of observations at time T
    :param series: array of float observation series
    :param slen: int estimate C_{t} at every time t module L in the cycle that the obs take on, 15 default
    :return: Python dict with series decomposition components
    """
    try:
        seasonal = {}
        season_averages = []
        n_seasons = int(len(series) / slen)
        for n in range(n_seasons):
            season_averages.append(
                sum(series[slen * n : slen * n + slen]) / float(slen)
            )
        for i in range(slen):
            sum_val_over_avg = 0.0
            for j in range(n_seasons):
                sum_val_over_avg += series[slen * j + i] - season_averages[j]
            seasonal[i] = sum_val_over_avg / n_seasons
        return {"season": seasonal, "season_avg": season_averages}
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on initial_seasonal_components() method: {ae}"
        )


@lru_cache(maxsize=30)
def triple_exp_smooth(
    series: List[float],
    n_pred: int = 10,
    slen: int = 15,
    gamma: float = 0.05,
    alpha: float = 0.05,
    beta: float = 0.005,
):
    """
    Compute triple exponential smooth series (Holt Winters) with alpha data smooth factor,
    beta trend smooth factor and gamma seasonal change smoothing factor.
    https://en.wikipedia.org/wiki/Exponential_smoothing
    :param series: array of float observation series
    :param n_pred: int n new estimation of value X_{t+m} based on the raw data, 10 set as default
    :param slen: int estimate C_{t} at every time t module L in the cycle that the obs take on, 15 default
    :param alpha: float alpha data smooth factor 0.05 set as default
    :param beta: float beta trend smooth factor 0.005 set as default
    :param gamma: float gamma seasonal change smoothing factor 0.05 set as default
    :return: Python dict with series decomposition components
    """
    try:
        result = []
        smooth = series[0]
        trend = __initial_trend(series, slen)
        seasonal = initial_seasonal_components(series, slen)["season"]
        for i in range(len(series) + n_pred):
            if i == 0:
                result.append(series[0])
                continue
            elif i >= len(series):
                m = i - len(series) + 1
                result.append((smooth + m * trend) + seasonal[i % slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha * (val - seasonal[i % slen]) + (
                    1 - alpha
                ) * (smooth + trend)
                trend = beta * (smooth - last_smooth) + (1 - beta) * trend
                seasonal[i % slen] = (
                    gamma * (val - smooth) + (1 - gamma) * seasonal[i % slen]
                )
                result.append(smooth + trend + seasonal[i % slen])
        delta = delta_series(result)
        return {
            "triple_exp_smooth": result,
            "trend": trend,
            "season": seasonal,
            "smooth": smooth,
            "delta": delta,
        }
    except ArithmeticError as ae:
        raise OphelianUtilitiesException(
            f"An error occurred on triple_exp_smooth() method: {ae}"
        )


def row_indexing(df, sort_by: str, is_desc=True):
    """
    Adds the index of every row as a new column for a given DataFrame
    :param df: df, data to index
    :param sort_by: str, name of the column to sort
    :param is_desc: bool, indicate if you need the data sorted in ascending or descending order
    :return: DataFrame
    """
    try:
        data = df.orderBy(desc(sort_by)) if is_desc else df.orderBy(asc(sort_by))
        field_name = sort_by.split("_")[0] + "_index"
        schema = StructType(
            data.schema.fields
            + [StructField(name=field_name, dataType=LongType(), nullable=False)]
        )
        logger.info("Indexing Row RDD")
        return (
            data.rdd.zipWithIndex()
            .map(lambda row: Row(*list(row[0]) + [row[1]]))
            .toDF(schema)
        )
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on row_indexing() method: {e}"
        )


def __list_match_sign(lst):
    return [
        sign
        for sign in lst
        if sign == "=="
        or sign == "!="
        or sign == ">="
        or sign == "<="
        or sign == ">"
        or sign == "<"
    ]


def __spark_condition_col_dict(w_col, c_col):
    return {
        "==": col(w_col) == c_col,
        "!=": col(w_col) != c_col,
        ">=": col(w_col) >= c_col,
        "<=": col(w_col) <= c_col,
        ">": col(w_col) > c_col,
        "<": col(w_col) < c_col,
    }


def string_match(string_condition: str):
    """
    String match will interpret any string sql syntax match case
    :param string_condition: str, name of the column to sort
    :return: Python dict with spark row match condition
    """
    try:
        str_split = string_condition.split(" ")
        match_op = __list_match_sign(str_split)
        search_index_condition_sign = [str_split.index(op) for op in match_op]
        where_col = str_split[search_index_condition_sign[0] - 1]
        condition = str_split[search_index_condition_sign[0] + 1]
        return __spark_condition_col_dict(where_col, condition)[match_op[0]]
    except Exception as e:
        raise OphelianUtilitiesException(
            f"An error occurred on string_match() method: {e}"
        )
