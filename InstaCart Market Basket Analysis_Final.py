#Importing all the libraries
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel


# Reading The data from 4 datsets into RDD's
priors1=sc.textFile('/FileStore/tables/hqoenphm1501873649307/order_products__prior.csv', use_unicode=False)
h1 = priors1.first()
priors = priors1.filter(lambda(row) : (row != h1)).map(lambda(r):(r.split(','))).map(lambda(orderID,productID,cart_order,reordered):(int(orderID),int(productID),int(cart_order),int(reordered)))

train1=sc.textFile('/FileStore/tables/hqoenphm1501873649307/order_products__train.csv', use_unicode=False)
h2 = train1.first()
train = train1.filter(lambda(row) : (row != h2)).map(lambda(r):(r.split(','))).map(lambda(orderID,productID,cart_order,reordered):(int(orderID),int(productID),int(cart_order),int(reordered)))

orders1=sc.textFile('/FileStore/tables/hqoenphm1501873649307/orders.csv', use_unicode=False)
h3 = orders1.first()
orders = orders1.filter(lambda(row) : (row != h3)).map(lambda(r):(r.split(','))).map(lambda(orderID,userID,eval_set,order_number,order_dow,order_hod,days_since):(int(orderID),int(userID),0 if eval_set == "prior" else 1 if eval_set=="train" else 2,int(order_number),int(order_dow),int(order_hod),0 if days_since=="" else int(days_since)))

products1=spark.read.csv("/FileStore/tables/hqoenphm1501873649307/products.csv", header=True, mode="DROPMALFORMED").rdd
h4 = products1.first()
products = products1.filter(lambda(row) : (row != h4)).map(lambda(row):(int(row[0]),int(row[2]),int(row[3])))


#Taking 10000 orders as the Dataset and filtering based on order type
orders_limited=sc.parallelize(orders.take(10000))
orders_prior = orders_limited.filter(lambda(row) : (row[2] == 0))
orders_train = orders_limited.filter(lambda(row) : (row[2] == 1))
orders_test = orders_limited.filter(lambda(row) : (row[2] == 2))

orders_prior.count()

orders_train.count()

orders_test.count()

#Getting all the prior_order order id's as Keys
prior_keys = orders_prior.keys().collect()

#Getting all the train_order order id's as Keys
train_keys = orders_train.keys().collect()

#Filtering the order related products based on keys
limited_prior_products = priors.filter(lambda(row) : (row[0] in prior_keys))
limited_train_products = train.filter(lambda(row) : (row[0] in train_keys))


#Converting the dataset to Pandas dataframe
ordersDF = orders_limited.toDF().toPandas()
ordersDF.columns=['orderID','userID','eval_set','order_number','order_dow','order_hod','days_since']
prior_productsDF=limited_prior_products.toDF().toPandas()
prior_productsDF.columns=['orderID','productID','cart_order','reordered']
train_productsDF=limited_train_products.toDF().toPandas()
train_productsDF.columns=['orderID','productID','cart_order','reordered']

products_DF=products.toDF().toPandas()
products_DF.columns=['productID','aisle_id','dep_id']


#Generating new product specific features for each productID
products = pd.DataFrame()
products['numorders'] = prior_productsDF.groupby(prior_productsDF.productID).size().astype(np.int32)
products['numreorders'] = prior_productsDF['reordered'].groupby(prior_productsDF.productID).sum().astype(np.float32)
products['reorder_ratio'] = (products.numreorders / products.numorders).astype(np.float32)
products_DF = products_DF.join(products, on='productID',how="inner")
products_DF.set_index('productID', drop=False, inplace=True)

products_DF

products_DF.productID.size


#Joining the new product related features to the Orders dataset
ordersDF.set_index('orderID', inplace=True, drop=False)
prior_productsDF = prior_productsDF.join(ordersDF, on='orderID', rsuffix='*', how="inner")
prior_productsDF.drop('orderID*', inplace=True, axis=1)

prior_productsDF

#Generating new features based on user's previous ordering history
userOrderData = pd.DataFrame()
userOrderData['avg_orderingGap'] = ordersDF.groupby('userID')['days_since'].mean().astype(np.float32)
userOrderData['total_orders'] = ordersDF.groupby('userID').size().astype(np.int16)

userOrderData


#Generating new features a user's previous order related products
userPriorData = pd.DataFrame()
userPriorData['items_purchased_total'] = prior_productsDF.groupby('userID').size().astype(np.int16)
userPriorData['all_items'] = prior_productsDF.groupby('userID')['productID'].apply(set)
userPriorData['distinct_items'] = (userPriorData.all_items.map(len)).astype(np.int16)

userPriorData

#Final user related data
usersFinalData = userPriorData.join(userOrderData)
usersFinalData['average_items'] = (usersFinalData.items_purchased_total / usersFinalData.total_orders).astype(np.float32)

usersFinalData

#Generating some more features related to products
userRelatedProductsData = prior_productsDF.copy()
userRelatedProductsData['userProduct'] = userRelatedProductsData.productID + userRelatedProductsData.userID * 100000
userRelatedProductsData = userRelatedProductsData.sort_values('order_number')
userRelatedProductsData = userRelatedProductsData.groupby('userProduct', sort=False).agg({'orderID': ['size', 'last'], 'cart_order': 'sum'})
userRelatedProductsData.columns = ['numorders', 'final_orderId', 'sum_cartOrder']
userRelatedProductsData.numorders = userRelatedProductsData.numorders.astype(np.int16)
userRelatedProductsData.sum_cartOrder = userRelatedProductsData.sum_cartOrder.astype(np.int16)
userRelatedProductsData.final_orderId = userRelatedProductsData.final_orderId.astype(np.int32)


userRelatedProductsData


train_productsDF.set_index(['orderID', 'productID'], inplace=True, drop=False)


#Function to transform the train and test datasets using the different features already created
def datasetFeatures(orderset):
    orders = []
    products = []
    ord_prod_labels = []
    for tuple1 in orderset.itertuples():
        orderID = tuple1.orderID
        userID = tuple1.userID
        userItems = usersFinalData.all_items[userID]
        products += userItems
        orders += [orderID] * len(userItems)
        ord_prod_labels += [(orderID, productID) in train_productsDF.index for productID in userItems]
        
    finalDF = pd.DataFrame({'orderID':orders, 'productID':products}, dtype=np.int32) 
    finalDF['label'] = ord_prod_labels
    finalDF['userID'] = finalDF.orderID.map(ordersDF.userID)
    finalDF['user_SumOrders'] = finalDF.userID.map(usersFinalData.total_orders)
    finalDF['user_items_purchased_total'] = finalDF.userID.map(usersFinalData.items_purchased_total)
    finalDF['distinct_items'] = finalDF.userID.map(usersFinalData.distinct_items)
    finalDF['user_avg_orderingGap'] = finalDF.userID.map(usersFinalData.avg_orderingGap)
    finalDF['user_average_items'] =  finalDF.userID.map(usersFinalData.average_items)
    finalDF['order_dow'] = finalDF.orderID.map(ordersDF.order_dow)
    finalDF['order_hod'] = finalDF.orderID.map(ordersDF.order_hod)
    finalDF['days_since'] = finalDF.orderID.map(ordersDF.days_since)
    finalDF['days_sinceProportion'] = finalDF.days_since / finalDF.user_avg_orderingGap
    finalDF['numorders'] = finalDF.productID.map(products_DF.numorders).astype(np.float32)
    finalDF['numreorders'] = finalDF.productID.map(products_DF.numreorders)
    finalDF['prod_reorderProportion'] = finalDF.productID.map(products_DF.reorder_ratio)
    finalDF['userProductGrp'] = finalDF.userID * 100000 + finalDF.productID
    finalDF['userProduct_orders'] = finalDF.userProductGrp.map(userRelatedProductsData.numorders)
    finalDF['userProduct_ordersRatio'] = (finalDF.userProduct_orders / finalDF.user_SumOrders).astype(np.float32)
    finalDF['userProduct_final_orderId'] = finalDF.userProductGrp.map(userRelatedProductsData.final_orderId)
    finalDF['userProduct_average_cartOrder'] = (finalDF.userProductGrp.map(userRelatedProductsData.sum_cartOrder) / finalDF.userProduct_orders).astype(np.float32)
    finalDF['userProduct_reorderProportion'] = (finalDF.userProduct_orders / finalDF.user_SumOrders).astype(np.float32)
    finalDF['userProduct_orders_sinceFinal'] = finalDF.user_SumOrders -finalDF.userProduct_final_orderId.map(ordersDF.order_number)
    return (finalDF)


ordersTest = ordersDF[ordersDF.eval_set == 2]
ordersTrain = ordersDF[ordersDF.eval_set == 1]


trainSet = datasetFeatures(ordersTrain)
testSet = datasetFeatures(ordersTest)


trainSet


trainSet.label = trainSet.label.astype(int)
testSet.label = testSet.label.astype(int)


#Features to be used for training the model
features =['label','user_SumOrders', 'user_items_purchased_total', 'distinct_items',
       'user_avg_orderingGap', 'user_average_items','order_dow',
       'order_hod', 'days_since', 'days_sinceProportion','numorders', 'numreorders',
       'prod_reorderProportion', 'userProduct_orders', 'userProduct_ordersRatio',
       'userProduct_average_cartOrder', 'userProduct_reorderProportion', 'userProduct_orders_sinceFinal']


trainSet[features]


testSet[features]


trainRDD = spark.createDataFrame(trainSet[features]).rdd
testRDD = spark.createDataFrame(testSet[features]).rdd


#Transforming the datasets to LibSVM format required for the regression model
trainFinal = trainRDD.map(lambda line: LabeledPoint(line[0],[line[1:]]))
testFinal = testRDD.map(lambda line: LabeledPoint(line[0],[line[1:]]))


trainFinal.count()


testFinal.count()


testFinal.collect()


#For Getting the threshold limit, Using Train dataset

(training1, training2) = trainFinal.randomSplit([0.7, 0.3])

training1.collect()


model_1 = RandomForest.trainRegressor(training1, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)
model_2 = GradientBoostedTrees.trainRegressor(training1,
                                            categoricalFeaturesInfo={}, numIterations=3)
model_3 = DecisionTree.trainRegressor(training1, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=5, maxBins=32)


predictionsRFTrain = model_1.predict(training1.map(lambda x: x.features))
predictionsGBTTrain = model_2.predict(training1.map(lambda x: x.features))
predictionsDTTrain = model_3.predict(training1.map(lambda x: x.features))

predictionsRFTrain.collect()

predictionsGBTTrain.collect()

predictionsDTTrain.collect()

training1.collect()


#We will use 0.19 as a threhold value to check whether to include a product in test order or not
labelsAndPredictionsRF = training1.map(lambda row: row.label).zip(predictionsRFTrain).map(lambda(row):(row[0],(0.0 if row[1] < 0.19 else 1.0)))
labelsAndPredictionsGBT = training1.map(lambda row: row.label).zip(predictionsGBTTrain).map(lambda(row):(row[0],(0.0 if row[1] < 0.19 else 1.0)))
labelsAndPredictionsDT = training1.map(lambda row: row.label).zip(predictionsDTTrain).map(lambda(row):(row[0],(0.0 if row[1] < 0.19 else 1.0)))


metricsRF = MulticlassMetrics(labelsAndPredictionsRF)
metricsGBT = MulticlassMetrics(labelsAndPredictionsGBT)
metricsDT = MulticlassMetrics(labelsAndPredictionsDT)


precisionRF = metricsRF.precision()
recallRF = metricsRF.recall()
f1ScoreRF = metricsRF.fMeasure()
print("RF summary on Train")
print("Precision = %s" % precisionRF)
print("Recall = %s" % recallRF)
print("F1 Score = %s" % f1ScoreRF)


precisionGBT = metricsGBT.precision()
recallGBT = metricsGBT.recall()
f1ScoreGBT = metricsGBT.fMeasure()
print("GBT summary on Train")
print("Precision = %s" % precisionGBT)
print("Recall = %s" % recallGBT)
print("F1 Score = %s" % f1ScoreGBT)


precisionDT = metricsDT.precision()
recallDT = metricsDT.recall()
f1ScoreDT = metricsDT.fMeasure()
print("DT summary on Train")
print("Precision = %s" % precisionDT)
print("Recall = %s" % recallDT)
print("F1 Score = %s" % f1ScoreDT)


#Training the RandomForest Regression model
model1 = RandomForest.trainRegressor(trainFinal, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)

predictionsRF = model1.predict(testFinal.map(lambda r: r.features))


predictionsRF.collect()


predictionsRF.count()


testSet['predictionRF']=predictionsRF.map(lambda r: (r, )).toDF().toPandas()


#Creating list of productID's for test orders based on the prediction
limit = 0.19
dat = dict()
for tuple1 in testSet.itertuples():
    if tuple1.predictionRF > limit:
        try:
            dat[tuple1.orderID] += ' ' + str(tuple1.productID)
        except:
            dat[tuple1.orderID] = str(tuple1.productID)


#Creating the final submission format of test orderID with its predicted products
for ord_id in ordersTest.orderID:
    if ord_id not in dat:
        dat[ord_id] = 'None'

resultRF = pd.DataFrame.from_dict(dat, orient='index')

resultRF.reset_index(inplace=True)
resultRF.columns = ['orderID', 'products']


print(resultRF)


#Training the GradientBoosted Regression model
model2 = GradientBoostedTrees.trainRegressor(trainFinal,
                                            categoricalFeaturesInfo={}, numIterations=3)


predictionsGBT = model2.predict(testFinal.map(lambda r: r.features))


predictionsGBT.collect()


testSet['predictionGBT']=predictionsGBT.map(lambda r: (r, )).toDF().toPandas()


#Creating list of productID's for test orders based on the prediction
limit = 0.19
dat2 = dict()
for tuple1 in testSet.itertuples():
    if tuple1.predictionGBT > limit:
        try:
            dat2[tuple1.orderID] += ' ' + str(tuple1.productID)
        except:
            dat2[tuple1.orderID] = str(tuple1.productID)


#Creating the final submission format of test orderID with its predicted products
for ord_id in ordersTest.orderID:
    if ord_id not in dat2:
        dat2[ord_id] = 'None'

resultGBT = pd.DataFrame.from_dict(dat2, orient='index')

resultGBT.reset_index(inplace=True)
resultGBT.columns = ['orderID', 'products']


print(resultGBT)


#Training the Decision Tree Regression model
model3 = DecisionTree.trainRegressor(trainFinal, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=5, maxBins=32)


predictionsDT = model3.predict(testFinal.map(lambda r: r.features))


predictionsDT.collect()


testSet['predictionsDT']=predictionsDT.map(lambda r: (r, )).toDF().toPandas()


#Creating list of productID's for test orders based on the prediction
limit = 0.19
dat3 = dict()
for tuple1 in testSet.itertuples():
    if tuple1.predictionsDT > limit:
        try:
            dat3[tuple1.orderID] += ' ' + str(tuple1.productID)
        except:
            dat3[tuple1.orderID] = str(tuple1.productID)


#Creating the final submission format of test orderID with its predicted products
for ord_id in ordersTest.orderID:
    if ord_id not in dat3:
        dat3[ord_id] = 'None'

resultDT = pd.DataFrame.from_dict(dat2, orient='index')

resultDT.reset_index(inplace=True)
resultDT.columns = ['orderID', 'products']


print(resultDT)



