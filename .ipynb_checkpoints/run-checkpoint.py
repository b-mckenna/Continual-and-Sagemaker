from sagemaker.sklearn import SKLearn
from sagemaker import estimator
import sagemaker
import boto3
import time
import pandas as pd
import os

# Create SageMaker session
sagemaker_session = sagemaker.Session()

role = "arn:aws:iam::765936207681:role/service-role/AmazonSageMaker-ExecutionRole-20191217T155901"

s3 = boto3.client('s3')

continual_api_key = os.environ.get("CONTINUAL_API_KEY", None)
run_id = os.environ.get("CONTINUAL_RUN_ID", None)

alpha = 0.1

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
    environment={"CONTINUAL_APIKEY": continual_api_key, "CONTINUAL_RUN_ID": run_id},
    hyperparameters={'alpha': alpha}
)

train_sklearn = "sklearn-training-job-{}".format(int(time.time()))

estimator.fit(job_name=train_sklearn)
