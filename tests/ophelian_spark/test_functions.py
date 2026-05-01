import unittest

from pyspark.sql import SparkSession

from ophelian.ophelian_spark.functions import Reshape


class TestNarrowFormat(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSession.builder.appName("test_narrow_format").getOrCreate()
        self.df = self.spark.createDataFrame(
            [(1, "a", 1), (1, "b", 2), (2, "a", 3), (2, "b", 4)],
            ["id", "letter", "number"],
        )

    def test_narrow_format(self):
        expected_df = self.spark.createDataFrame(
            [
                (1, "letter", "a"),
                (1, "number", "1"),
                (1, "letter", "b"),
                (1, "number", "2"),
                (2, "letter", "a"),
                (2, "number", "3"),
                (2, "letter", "b"),
                (2, "number", "4"),
            ],
            ["id", "pivot", "value"],
        )

        result_df = Reshape.narrow_format(self.df, "id", ["pivot", "value"])
        self.assertEqual(expected_df.collect(), result_df.collect())
        self.assertIsInstance(result_df, Reshape)

    def tearDown(self):
        self.spark.stop()


class TestWideFormat(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSession.builder.appName("test_narrow_format").getOrCreate()
        self.df = self.spark.createDataFrame(
            [
                (1, "letter", "a"),
                (1, "number", "1"),
                (1, "letter", "b"),
                (1, "number", "2"),
                (2, "letter", "a"),
                (2, "number", "3"),
                (2, "letter", "b"),
                (2, "number", "4"),
            ],
            ["id", "pivot", "value"],
        )

    def test_narrow_format(self):
        expected_df = self.spark.createDataFrame(
            [(1, 0.0, 1.5), (2, 0.0, 3.5)], ["id", "letter", "number"]
        )

        result_df = Reshape.wide_format(self.df, "id", "pivot", {"value": "mean"})
        self.assertEqual(expected_df.collect(), result_df.collect())
        self.assertIsInstance(result_df, Reshape)

    def tearDown(self):
        self.spark.stop()


if __name__ == "__main__":
    unittest.main()
