from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import broadcast
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pandas as pd
import sys

# Create class
class ALS_prediction(object):

	def __init__(self):
		self.model = ALSModel.load("als_explicit_collab")
		# Import the dataset
		collab = pd.read_csv(sys.argv[1])
		collab_users = collab.drop(columns=['interactionTime', 'streamerId'], axis=1)
		collab_users = collab_users.rename(columns={'userId': 'user_id'})

		collab_items = collab.drop(columns=['interactionTime', 'userId'], axis=1)
		collab_items = collab_items.rename(columns={'userId': 'user_id'})
		collab_items = collab.rename(columns={'streamerId':'item_id'})
		# Create into spark dataframe and split
		conf = SparkConf().setAppName('app').setMaster('local[*]')
		sc = SparkContext(conf=conf)
		sql_context = SQLContext(sc)

		user_df = sql_context.createDataFrame(collab_users)
		item_df = sql_context.createDataFrame(collab_items)

		prediction_data = broadcast(item_df).crossJoin(user_df).select('user_id', 'item_id').rdd.map(tuple)

		self.all_ratings = self.model.predictAll(prediction_data)

	# Function to take user_id, item_id and return the prediction
	def predict(self, uid, iid):
		return self.all_ratings.select("rating").where("user_id == '"+str(uid)+"' and item_id == '"+str(iid)+"'").collect()

		

def main():
	ALS_prediction().predict(1,1)

main()
