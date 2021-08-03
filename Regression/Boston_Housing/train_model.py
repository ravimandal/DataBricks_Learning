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

#create A and y
X=boston_df.drop(['Target'],axis=1)
y=boston_df['Target']

# COMMAND ----------

#creating data for model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=123)

# COMMAND ----------

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='linear_regression'):
  model = LinearRegression()
  model.fit(X_train, y_train)
 
  predictions_test = model.predict(X_test)
  mae_score =sklm.mean_absolute_error(y_test, predictions_test)
  # Use MAE as metrics.
  mlflow.log_metric('MAE', mae_score)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, model.predict(X_train))
  
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.sklearn.log_model(model,"linear_regression", conda_env=conda_env, signature=signature)

# COMMAND ----------

import mlflow
logged_model = 'runs:/8dc10c33a8a34c30ae43575d9dddfdac/linear_regression'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
pred=loaded_model.predict(pd.DataFrame(boston_df))
