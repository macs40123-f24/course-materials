from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinHashLSH

spark = SparkSession \
        .builder \
        .appName("MinHashLSH") \
        .getOrCreate()

# fit model on sparse vectors of customer purchases
# e.g. customer 0 purchased items 0, 1, 2;
# customer 1 purchased items 2, 3, 4; etc.
data = [(0, Vectors.sparse(6, [0, 1, 2], [1.0, 1.0, 1.0]),),
        (1, Vectors.sparse(6, [2, 3, 4], [1.0, 1.0, 1.0]),),
        (2, Vectors.sparse(6, [0, 2, 4], [1.0, 1.0, 1.0]),)]

df = spark.createDataFrame(data, ["id", "features"])

# Instantiate model
mh = MinHashLSH()
mh.setInputCol("features")
mh.setOutputCol("hashes")
mh.setSeed(12345)

# fit model based on customer purchase data in df
model = mh.fit(df)
model.setInputCol("features")

# assume we collect additional customer purchase data: df2
data2 = [(3, Vectors.sparse(6, [1, 3, 5], [1.0, 1.0, 1.0]),),
         (4, Vectors.sparse(6, [2, 3, 5], [1.0, 1.0, 1.0]),),
         (5, Vectors.sparse(6, [1, 2, 4], [1.0, 1.0, 1.0]),)]

df2 = spark.createDataFrame(data2, ["id", "features"])

# identify n customers in df2 most indexically similar to customer_a
n = 1
customer_a = Vectors.sparse(6, [1, 2], [1.0, 1.0])
model.approxNearestNeighbors(df2, customer_a, n) \
     .select(
         col("id"),
         col("features"),
         col("distCol")) \
     .show()

# find all customers in df < 0.6 jaccard distance from customers in df2 
model.approxSimilarityJoin(df, df2, 0.6, distCol="JaccardDistance") \
     .select(
         col("datasetA.id").alias("idA"),
         col("datasetB.id").alias("idB"),
         col("JaccardDistance")) \
     .show()
