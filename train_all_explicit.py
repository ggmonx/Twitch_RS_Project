from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
pbounds = {'regParam': (0,1),
            'rank': (1,30)}





# Import the dataset
df = pd.read_csv('data\collab\collab_filter.csv')
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
#iteration = 10 if len(sys.argv) < 2 else int(sys.argv[1])

def findOptimalModel(regParam=0.3, rank=10):
    rank = int(rank)
    als = ALS(maxIter=40, regParam=regParam, rank=rank, nonnegative=True, userCol='user_id', itemCol="item_id", ratingCol="rating", coldStartStrategy='drop')
    lrModel = als.fit(training)
    predictions = lrModel.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol='rating',
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return -1 *rmse # bayes opt maximizes function

def trainOptimalModel(regParam=0.3, rank=10):
    rank = int(rank)
    als = ALS(maxIter=40, regParam=regParam, rank=rank, nonnegative=True, userCol='user_id', itemCol="item_id", ratingCol="rating", coldStartStrategy='drop')
    lrModel = als.fit(training)
    return lrModel

optimizer = BayesianOptimization(
    f=findOptimalModel,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1
)

optimizer.probe( # default values for als
    params={'regParam': .001,
            'rank': 10},
    lazy=True,
)
optimizer.maximize(
    init_points=5,
    n_iter=20
)

print("Max params", optimizer.max)

rmses=[]
iterations= []
for i, res in enumerate(optimizer.res):
    iterations.append(i)
    rmses.append(-1 * res['target'])

plt.plot(iterations, rmses)
plt.ylabel('rmse scores')
plt.xlabel('Iteration Number')
plt.title('Bayesian Optimization for ALS')
plt.show()
