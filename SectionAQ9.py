
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Spark').getOrCreate()

df = spark.read.format("com.databricks.spark.csv").option("header", "true").load("flights.csv")

df.registerTempTable("df")

ds= spark.sql("select Carrier,Sum(DepDelay) AS Total_DEP_Delays from df Group By Carrier")

dataset= ds.toPandas()

dataset.plot(kind='bar',figsize=(10, 5),x='Carrier',y='Total_DEP_Delays', color='red')

print(dataset)
plt.show()



