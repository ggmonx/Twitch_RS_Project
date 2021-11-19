from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pandas as pd
import sys
import os

os.environ['HADOOP_HOME'] = "C:/Hadoop"
sys.path.append("C:/Hadoop/bin")


# Import the dataset
df = pd.read_csv('data\collab_filter_100k_new_item_index.csv')
collab = df.copy()#drop(columns=['streamId'])
collab = collab.rename(columns={'streamerId':'item_id','interactionTime':'rating', 'userId': 'user_id'})
collab.head().info()

# Create into spark dataframe and split
conf = SparkConf().setAppName('app').set("spark.driver.memory", "16g").setMaster('local[4]')
sc = SparkContext(conf=conf)

# Set your own checkpoint directory on your system
sc.setCheckpointDir('C:/Users/Do-While/Desktop/256_term_project/check_point_directory/als')
sql_context = SQLContext(sc)

ratings = sql_context.createDataFrame(collab)
(training, test) = ratings.randomSplit([0.8,0.2], 38)

# Grab argument for number of iterations to run
iteration = 10 if len(sys.argv) < 2 else int(sys.argv[1])

# Build model using ALS - Using explicit ratings
als = ALS(maxIter=iteration, regParam=0.01, userCol='user_id', itemCol="item_id", ratingCol="rating", coldStartStrategy='drop')

# Train the model
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Save the model
model.save("models/als_explicit_collab")
print("Model successfully saved")
