from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession \
        .builder \
        .appName("frequent_itemsets") \
        .getOrCreate()

# Read Instacart order data and group products into 'baskets' by order ID
instacart_orders = \
    spark.read.csv('/project/macs40123/order_products_prior.csv', header=True)
instacart_baskets = \
    instacart_orders.groupBy('order_id') \
                    .agg(F.collect_list('product_id').alias('basket'))

# Fit FPGrowth model (support = 3.21m * 0.001 = 3,210)
fp = FPGrowth(minConfidence=0.5, minSupport=0.001)
fpm = fp.fit(instacart_baskets.select(instacart_baskets.basket.alias('items')))

# Find frequent itemsets and write to file
freq_itemsets = fpm.freqItemsets \
                   .sort("freq", ascending=False)
freq_itemsets.write.mode('overwrite').parquet('freq_itemsets')

# Find association rules and write to file
association_rules = fpm.associationRules \
                       .sort("antecedent", "consequent")
association_rules.write.mode('overwrite').parquet('association_rules')