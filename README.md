<div align="center" style="padding: 20px;">
  <img src="docs/img/ophelian-ai-sticker.png" alt="Ophelian AI Sticker" width="200" style="margin-top: 20px;">

  <p style="color: #FFF; font-family: 'Helvetica Neue', Arial, sans-serif; text-align: center; max-width: 600px; margin: 20px auto; font-weight: bold;">
  </p>
  
  <div style="margin-top: 20px;">
    <a href="https://pypi.org/project/ophelian/" style="margin-right: 10px;">
      <img src="https://img.shields.io/pypi/v/ophelian.svg" alt="PyPI">
    </a>
    <a href="https://hub.docker.com/r/luisfalva/ophelian" style="margin-right: 10px;">
      <img src="https://img.shields.io/docker/v/luisfalva/ophelian?sort=semver" alt="Docker Hub">
    </a>
    <a href="https://github.com/LuisFalva/ophelia/actions/workflows/release.yml" style="margin-right: 10px;">
      <img src="https://github.com/LuisFalva/ophelia/actions/workflows/release.yml/badge.svg" alt="Release Build Status">
    </a>
    <a href="https://github.com/LuisFalva/ophelia/actions/workflows/docker-image.yml" style="margin-right: 10px;">
      <img src="https://github.com/LuisFalva/ophelia/actions/workflows/docker-image.yml/badge.svg" alt="Docker Image Build Status">
    </a>
    <a href="https://ophelian.readme.io/">
      <img src="https://img.shields.io/badge/docs-Documentation-orange.svg" alt="Docs">
    </a>
  </div>
</div>

# Ophelian On Mars

---

'*Ophelian On Mars*' 👽, the ultimate destination for ML, Data Science, and AI professionals. Your go-to framework for seamlessly putting ML prototypes into production—where everyone wants to be, but only a few succeed.

# 🚀 Motivations

As data professionals, we aim to minimize the time spent deciphering the intricacies of PySpark's framework. Often, we seek a straightforward, Pandas-style approach to compute tasks without delving into highly optimized Spark code. 

To address this need, Ophelian was created with the following goals:

- **Simplicity**: Provide a simple and intuitive way to perform data computations, emulating the ease of Pandas.
- **Efficiency**: Wrap common patterns for data extraction and transformation in a single entry function that ensures Spark-optimized performance.
- **Code Reduction**: Significantly reduce the amount of code required by leveraging a set of Spark optimization techniques for query execution.
- **Streamlined ML Pipelines**: Facilitate the lifecycle of any PySpark ML pipeline by incorporating optimized methods and reducing redundant coding efforts.

By focusing on these motivations, Ophelian aims to enhance productivity and efficiency for data engineers and scientists, allowing them to concentrate on their core tasks without worrying about underlying Spark optimizations.

# 📝 Generalized ML Features

Ophelian focuses on creating robust and efficient machine learning (ML) pipelines, making them easily replicable and secure for various ML tasks. Key features include optimized techniques for handling data skewness, user-friendly interfaces for building custom models, and streamlined data mining pipelines with Ophelian pipeline wrappers. Additionally, it functions as an emulator of NumPy and Pandas, offering similar functionalities for a seamless user experience. Below are the detailed features:

- **Framework for Building ML Pipelines**: Simplified and secure methods to construct ML pipelines using PySpark, ensuring replication and robustness.
- **Optimized Techniques for Data Skewness and Partitioning**: Embedded strategies to address and mitigate data skewness issues, improving model performance and accuracy.
- **Build Your Own Models (BYOM)**: User-friendly software for constructing custom models and data mining pipelines, leveraging frameworks like PySpark, Beam, Flink, PyTorch, and more, with Ophelian native wrappers for enhanced syntax flexibility and efficiency.
- **NumPy and Pandas Functionality Syntax Emulation**: Emulates the functions and features of NumPy and Pandas, making it intuitive and easy for users familiar with these libraries to transition and utilize similar functionalities within an ML pipeline.

These features empower users with the tools they need to handle complex ML tasks effectively, ensuring a seamless experience from data processing to model deployment. users with the tools they need to handle complex machine learning tasks effectively, ensuring a seamless experience from data processing to model deployment.

# Getting Started:

### 📜 Requirements

Before starting, you'll need to have installed: 
- pyspark >= 3.0.x
- pandas >= 1.1.3
- numpy >= 1.19.1
- dask >= 2.30.x
- scikit-learn >= 0.23.x

Additionally, if you want to use the Ophelian packages, you'll also need Python (supported 3.7 and 3.8 versions) and pip installed.

### 🛠 Install Ophelian pypi package

Just drop a pip install to the `Ophelian` pypi registry and import `Ophelian`:
```sh
pip install ophelian==0.1.4
```

### 📦 Importing and initializing Ophelian

To initialize `Ophelian` with Spark embedded session use:

```python
from ophelian.start import OphelianSession
ophelian = OphelianSession("Spark App Name")
sc = ophelian.Spark.build_spark_context()
  ____          _            _  _               
 / __ \        | |          | |(_)              
| |  | | _ __  | |__    ___ | | _   __ _  _ __  
| |  | || '_ \ | '_ \  / _ \| || | / _` || '_ \ 
| |__| || |_) || | | ||  __/| || || (_| || | | |
 \____/ | .__/ |_| |_| \___||_||_| \__,_||_| |_|
        | |                                     
        |_|                                     
  ____         
 / __ \        
| |  | | _ __  
| |  | || '_ \ 
| |__| || | | |
 \____/ |_| |_|       
               
 __  __                    _ 
|  \/  |                  | |
| \  / |  __ _  _ __  ___ | |
| |\/| | / _` || '__|/ __|| |
| |  | || (_| || |   \__ \|_|
|_|  |_| \__,_||_|   |___/(_)

```
Main class objects provided by initializing Ophelia session:

- `read` & `write`

```python
from ophelian.ophelian_spark.read.spark_read import Read
from ophelian.ophelian_spark.write.spark_write import Write
```
- `generic` & `functions`

```python
from ophelian.ophelian_spark.functions import (
  Shape, Rolling, Reshape, CorrMat, CrossTabular, 
  PctChange, Selects, DynamicSampling
)
from ophelian.ophelian_spark.generic import (
  split_date, row_index, lag_min_max_data, regex_expr, remove_duplicate_element,
  year_array, dates_index, sorted_date_list, feature_pick, binary_search,
  century_from_year, simple_average, delta_series, simple_moving_average, average,
  weight_moving_average, single_exp_smooth, double_exp_smooth, initial_seasonal_components,
  triple_exp_smooth, row_indexing, string_match
)
```
- ML package for `unsupervised`, `sampling` and `feature_miner` objects

```python
from ophelian.ophelian_spark.ml.sampling.synthetic_sample import SyntheticSample
from ophelian.ophelian_spark.ml.unsupervised.feature import PCAnalysis, SingularVD
from ophelian.ophelian_spark.ml.feature_miner import (
  BuildStringIndex, BuildOneHotEncoder, 
  BuildVectorAssembler, BuildStandardScaler,
  SparkToNumpy, NumpyToVector
)
```

Let me show you some application examples:

The `Read` class implements Spark reading object in multiple formats `{'csv', 'parquet', 'excel', 'json'}`

```python
from ophelian.ophelian_spark.read.spark_read import Read

spark_df = spark.readFile(path, 'csv', header=True, infer_schema=True)
```

Also, you may import class `Shape` from factory `functions` in order to see the dimension of our spark DataFrame such as numpy style.

```python
from ophelian.ophelian_spark.functions import Shape

dic = {
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Year': [2010, 2010, 2010, 2011, 2011, 2011, 2012, 2012, 2012],
    'Revenue': [100, 200, 300, 110, 190, 320, 120, 220, 350]
}
dic_to_df = spark.createDataFrame(pd.DataFrame(data=dic))
dic_to_df.show(10, False)

+---------+------------+-----------+
| Product |    Year    |  Revenue  |
+---------+------------+-----------+
|    A    |    2010    |    100    |
|    B    |    2010    |    200    |
|    C    |    2010    |    300    |
|    A    |    2011    |    110    |
|    B    |    2011    |    190    |
|    C    |    2011    |    320    |
|    A    |    2012    |    120    |
|    B    |    2012    |    220    |
|    C    |    2012    |    350    |
+---------+------------+-----------+

dic_to_df.Shape
(9, 3)
```

The `pct_change` wrapper is added to the Spark `DataFrame` class in order to have the most commonly used method in Pandas
objects to get the relative percentage change from one observation to another, sorted by a date-type column and lagged by a numeric-type column.

```python
from ophelian.ophelian_spark.functions import PctChange

dic_to_df.pctChange().show(10, False)

+---------------------+
|       Revenue       |
+---------------------+
| null                |
| 1.0                 |
| 0.5                 |
| -0.6333333333333333 |
| 0.7272727272727273  |
| 0.6842105263157894  |
| -0.625              |
| 0.8333333333333333  |
| 0.5909090909090908  |
+---------------------+
```

Another option is to configure all receiving parameters from the function, as follows:
- `periods`; this parameter will control the offset of the lag periods. Since the default value is 1, this will always return a lag-1 information DataFrame.
- `partition_by`; this parameter will fix the partition column over the DataFrame, e.g. 'bank_segment', 'assurance_product_type'.
- `order_by`; order by parameter will be the specific column to order the sequential observations, e.g. 'balance_date', 'trade_close_date', 'contract_date'.
- `pct_cols`; percentage change col (pct_cols) will be the specific column to lag-over giving back the relative change between one element to other, e.g. 𝑥𝑡 ÷ 𝑥𝑡 − 1

In this case, we will specify only the `periods` parameter to yield a lag of -2 days over the DataFrame.
```python
dic_to_df.pctChange(periods=2).na.fill(0).show(5, False)

+--------------------+
|Revenue             |
+--------------------+
|0.0                 |
|0.0                 |
|2.0                 |
|-0.44999999999999996|
|-0.3666666666666667 |
+--------------------+
only showing top 5 rows
```

Adding parameters: `partition_by`, `order_by` & `pct_cols`
```python
dic_to_df.pctChange(partition_by="Product", order_by="Year", pct_cols="Revenue").na.fill(0).show(5, False)

+---------------------+
|Revenue              |
+---------------------+
|0.0                  |
|-0.050000000000000044|
|0.1578947368421053   |
|0.0                  |
|0.06666666666666665  |
+---------------------+
only showing top 5 rows
```

You may also lag more than one column at a time by simply adding a list with string column names:
```python
dic_to_df.pctChange(partition_by="Product", order_by="Year", pct_cols=["Year", "Revenue"]).na.fill(0).show(5, False)

+--------------------+---------------------+
|Year                |Revenue              |
+--------------------+---------------------+
|0.0                 |0.0                  |
|4.975124378110429E-4|-0.050000000000000044|
|4.972650422674363E-4|0.1578947368421053   |
|0.0                 |0.0                  |
|4.975124378110429E-4|0.06666666666666665  |
+--------------------+---------------------+
only showing top 5 rows
```
 
## 🤔 Contributing to Ophelian On Mars

We welcome contributions from everyone! If you have an idea, a question, or if you've found a bug that needs fixing, please open an [issue ticket](https://github.com/LuisFalva/ophelian/issues).

You can find guidelines for submitting an issue request in our repository. Additionally, you can refer to the [Open Source Contribution Guide best practices](https://opensource.guide/) to get started.

## 📠 Support or Contact

Having trouble with Ophelian? Yo can DM me at [falvaluis@gmail.com](https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox?compose=CllgCJZZQVJHBJKmdjtXgzlrRcRktFLwFQsvWKqcTRtvQTVcHvgTNSxVzjZqjvDFhZlVJlPKqtg) and I’ll help you sort it out.

## 📃 License

Released under the Apache License, version 2.0.
