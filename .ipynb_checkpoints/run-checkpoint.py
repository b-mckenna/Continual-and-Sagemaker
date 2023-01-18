from sagemaker.sklearn import SKLearn
from sagemaker import estimator
import sagemaker
import boto3
import time
import pandas as pd
import os

# Create SageMaker session
sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

s3 = boto3.client('s3')

continual_api_key = 'apikey/4ca70a3a49c142f0a73be901a0b8bef8'

# Create estimator
estimator = SKLearn(
    entry_point='train.py',
    role=role,
    sagemaker_session=sagemaker_session,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='1.0-1',
    py_version='py3',
    source_dir='.', 
    environment={"CONTINUAL_APIKEY": continual_api_key}
)

train_sklearn = "sklearn-training-job-{}".format(int(time.time()))

estimator.fit(job_name=train_sklearn)
