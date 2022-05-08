#%%
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Spark').getOrCreate()

df = spark.read.format("com.databricks.spark.csv").option("header", "true").load("flights.csv")

df.registerTempTable("df")
print("As Hour of day is not present in dataset, i am answering for Day of month. /n")
print(spark.sql("select Cast(DayofMonth AS INT),Sum(DepDelay) AS Total_DEP_Delay from df Group By DayofMonth Order By DayofMonth").show(31))

# %%
