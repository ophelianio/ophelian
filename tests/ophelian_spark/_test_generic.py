import unittest

from pyspark.sql import SparkSession

from ophelian.ophelian_spark.generic import row_index, split_date, union_all


class GenericTest(unittest.TestCase):

    def __init__(self):
        super().__init__()
        self.spark = SparkSession.builder.appName("Generic Tests").getOrCreate()

    def test_union_all(self):
        # Create some test dataframes
        df1 = self.spark.createDataFrame([(1, 2), (3, 4)], ["col1", "col2"])
        df2 = self.spark.createDataFrame([(5, 6), (7, 8)], ["col1", "col2"])
        df3 = self.spark.createDataFrame([(9, 10), (11, 12)], ["col1", "col2"])

        # Call the union_all function with the test dataframes
        result = union_all([df1, df2, df3])

        # Verify that the returned dataframe has the correct number of rows
        self.assertEqual(result.count(), 6)

        # Verify that the returned dataframe has the correct schema
        self.assertEqual(result.schema, df1.schema)

        # Verify that the rows from all input dataframes are present in the result dataframe
        rows = result.collect()
        self.assertIn((1, 2), rows)
        self.assertIn((3, 4), rows)
        self.assertIn((5, 6), rows)
        self.assertIn((7, 8), rows)
        self.assertIn((9, 10), rows)
        self.assertIn((11, 12), rows)

    def test_split_date(self):
        # Create a test dataframe with a date column
        df = self.spark.createDataFrame(
            [(1, "2022-01-01"), (2, "2022-02-01")], ["col1", "col_date"]
        )

        # Call the split_date function with the test dataframe and date column
        result = split_date(df, "col_date")

        # Verify that the returned dataframe has the correct number of rows
        self.assertEqual(result.count(), 2)

        # Verify that the returned dataframe has the correct schema
        self.assertEqual(result.schema, df.schema)

        # Verify that the date column has been split into year, month, and day columns
        self.assertIn("col_date_year", result.columns)
        self.assertIn("col_date_month", result.columns)
        self.assertIn("col_date_day", result.columns)

        # Verify that the year, month, and day columns have the correct values
        rows = result.collect()
        self.assertEqual(rows[0]["col_date_year"], 2022)
        self.assertEqual(rows[0]["col_date_month"], 1)
        self.assertEqual(rows[0]["col_date_day"], 1)
        self.assertEqual(rows[1]["col_date_year"], 2022)
        self.assertEqual(rows[1]["col_date_month"], 2)
        self.assertEqual(rows[1]["col_date_day"], 1)

    def test_row_index(self):
        # Create a test dataframe with an order column
        df = self.spark.createDataFrame([(1, 3), (2, 2), (3, 1)], ["col1", "col_order"])

        # Call the row_index function with the test dataframe and order column
        result = row_index(df, "col_order")

        # Verify that the returned dataframe has the correct number of rows
        self.assertEqual(result.count(), 3)

        # Verify that the returned dataframe has the correct schema
        self.assertEqual(result.schema, df.schema)

        # Verify that the row_num column has been added
        self.assertIn("row_num", result.columns)

        # Verify that the row_num column has the correct values
        rows = result.collect()
        self.assertEqual(rows[0]["row_num"], 1)
        self.assertEqual(rows[1]["row_num"], 2)
        self.assertEqual(rows[2]["row_num"], 3)

        # Verify that the rows are ordered correctly
        self.assertEqual(rows[0]["col1"], 3)
        self.assertEqual(rows[1]["col1"], 2)
        self.assertEqual(rows[2]["col1"], 1)


if __name__ == "__main__":
    unittest.main()
