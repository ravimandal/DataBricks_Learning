# Databricks notebook source
#import libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import datasets

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# COMMAND ----------

# load data
boston = datasets.load_boston()
boston_df = pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df['Target']=boston.target

# COMMAND ----------

# import mlflow
# logged_model = 'runs:/8dc10c33a8a34c30ae43575d9dddfdac/linear_regression'

# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)

# # Predict on a Pandas DataFrame.
# import pandas as pd
# pred=loaded_model.predict(pd.DataFrame(boston_df))

# COMMAND ----------

# This code is the same as the last block of "Building a Baseline Model". No change is required for clients to get the new model!
model_name='boston_housing'
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# COMMAND ----------

#Running prediction on pandas dataframe
boston_df_new=boston_df.copy()
boston_df_new['prediction']=model.predict(boston_df)
display(boston_df_new)

# COMMAND ----------

# To simulate a new corpus of data, save the existing X_train data to a Delta table. 
# In the real world, this would be a new batch of data.
spark_df = spark.createDataFrame(boston_df)
# Replace <username> with your username before running this cell.
table_path = "dbfs:/<username>/delta/boston_data"
# Delete the contents of this path in case this cell has already been run
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

# MAGIC %fs 
# MAGIC ls dbfs:/<username>/delta/boston_data

# COMMAND ----------

import mlflow.pyfunc
 
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# Read the "new data" from Delta
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

from pyspark.sql.functions import struct
 
# Apply the model to the new data
udf_inputs = struct(*(['CRIM',
 'ZN',
 'INDUS',
 'CHAS',
 'NOX',
 'RM',
 'AGE',
 'DIS',
 'RAD',
 'TAX',
 'PTRATIO',
 'B',
 'LSTAT']))
 
new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

display(new_data)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS boston_housing
# MAGIC   USING DELTA
# MAGIC   LOCATION "dbfs:/<username>/delta/boston_data"

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from default.boston_housing

# COMMAND ----------

temp=spark.sql("""
select * from default.boston_housing
""")
display(temp)

# COMMAND ----------

temp=spark.read.format('delta').load(table_path)
display(temp)

# COMMAND ----------


