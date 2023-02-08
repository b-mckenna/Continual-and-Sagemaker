from sagemaker.sklearn import SKLearn
from sagemaker import estimator
import sagemaker
import boto3
import time
import pandas as pd
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer

with open('params.yaml') as config_file:
    config = yaml.safe_load(config_file)

# Create SageMaker session
sagemaker_session = sagemaker.Session()
s3 = boto3.client('s3')
role = config['environment']['sm_role']

# Read model params from yaml
alpha = config['train']['alpha']
max_depth = config['train']['max_depth']
eta = config['train']['eta']
num_class = config['train']['num_class']
eval_metric = config['train']['eval_metric']

# Read Continual params from OS
continual_api_key = os.environ.get("CONTINUAL_API_KEY", None)
run_id = os.environ.get("CONTINUAL_RUN_ID", None)

# Create estimator object
estimator = SKLearn(
    entry_point='train.py',
    role=role,
    sagemaker_session=sagemaker_session,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='1.0-1',
    py_version='py3',
    source_dir='.',
    environment={"CONTINUAL_API_KEY": continual_api_key, "CONTINUAL_RUN_ID": run_id},
    hyperparameters={'alpha': alpha, 'max_depth':max_depth, 'eta':eta, 'num_class':num_class, 'eval_metric':eval_metric}
)

train_sklearn = "sklearn-training-job-{}".format(int(time.time()))

# Training
estimator.fit(job_name=train_sklearn)

# Deploy real-time endpoint
model = estimator.create_model()
predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge',endpoint_name='prod-endpoint')

# For demo purposes, make sure to delete endpoint and endpoint configuration
predictor.delete_endpoint(endpoint_name='prod-endpoint')