from pyspark.sql.types import IntegerType, StringType, StructField, StructType

bank_schema = StructType(
    [
        StructField("age", IntegerType(), True),
        StructField("job", StringType(), True),
        StructField("marital", StringType(), True),
        StructField("education", StringType(), True),
        StructField("housing", StringType(), True),
        StructField("balance", IntegerType(), True),
        StructField("duration", IntegerType(), True),
        StructField("poutcome", StringType(), True),
    ]
)
