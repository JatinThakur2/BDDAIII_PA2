
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Spark').getOrCreate()

df = spark.read.format("com.databricks.spark.csv").option("header", "true").load("flights.csv")

df.registerTempTable("df")

print(spark.sql("select Carrier,AVG(DepDelay) AS Average_DEP_Delay from df Group By Carrier").show())

