from sagemaker.sklearn import SKLearn
from sagemaker import estimator
import sagemaker
import boto3
import time
import pandas as pd
import os
import yaml

with open('params.yaml') as config_file:
    config = yaml.safe_load(config_file)

# Create SageMaker session
sagemaker_session = sagemaker.Session()
s3 = boto3.client('s3')
role = config['environment']['sm_role']

# Model params
alpha = config['train']['alpha']
max_depth = config['train']['max_depth']
eta = config['train']['eta']
num_class = config['train']['num_class']
eval_metric = config['train']['eval_metric']

# Continual params
continual_api_key = os.environ.get("CONTINUAL_API_KEY", None)
run_id = os.environ.get("CONTINUAL_RUN_ID", None)

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
    hyperparameters={'alpha': alpha, 'max_depth':max_depth, 'eta':eta, 'num_class':num_class, 'eval_metric':eval_metric}
)

train_sklearn = "sklearn-training-job-{}".format(int(time.time()))

estimator.fit(job_name=train_sklearn)

# to create model
model = estimator.create_model()
# to deploy
predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge',endpoint_name='geneclassificationendpoint')

# to predict
print(predictor.predict("ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG"))
