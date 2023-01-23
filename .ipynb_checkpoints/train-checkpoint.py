from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import train_test_split
import boto3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
import pandas as pd
import tempfile
import joblib
from continual import Client
from continual.python.sdk.runs import Run
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_config(CONFIG_KEYS):
    config = {}
    for key in CONFIG_KEYS:
        config[key] = os.getenv(key)
    return config

def getKmers(sequence, size=7):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def transform(human_data):
	human_data['words']=human_data['sequence'].apply(lambda x: getKmers(x))
	human_data_revised=human_data.drop(columns='sequence',axis=1)
	human_texts = list(human_data_revised['words'])
	for item in range(len(human_texts)):
		human_texts[item] = ' '.join(human_texts[item])
		
	y=human_data_revised['class'].values
	cv = CountVectorizer(ngram_range=(4,4))
	X = cv.fit_transform(human_texts)
	return X, y

def setup_metrics_dict(accuracy, f1, recall, precision):
	# Create metrics
	metric_dicts = [
		dict(
			key="accuracy",
			value=accuracy,
			direction="HIGHER",
			group_name="test",
			step=1,
		),
		dict(
			key="precision",
			value=precision,
			direction="HIGHER",
			group_name="test",
			step=1,
		),
		dict(
			key="f1",
			value=f1,
			direction="HIGHER",
			group_name="test",
			step=1,
		),
		dict(
			key="recall",
			value=recall,
			direction="HIGHER",
			group_name="test",
			step=1,
		),
	]
	return metric_dicts

def calculate_metrics(y_test,y_pred):
	accuracy = accuracy_score(y_test,y_pred)
	precision=precision_score(y_test,y_pred,average='weighted')
	recall=recall_score(y_test,y_pred,average='weighted')
	f1=f1_score(y_test,y_pred,average='weighted')
	return setup_metrics_dict(accuracy, f1, recall, precision)

def create_confusion_matrix(name, y_test,y_pred):
    file_path = "%s/" %os.getcwd()+name+"_confusion_matrix.png"
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='viridis')
    plt.savefig(file_path)
    return file_path

def run_experiment(experiment, name, model, X_train, y_train, y_test):
	model.fit(X_train,y_train)

	joblib.dump(model, open(name+"_model", "wb"))

	# Test model
	y_pred=model.predict(X_test)

	metrics_dict = calculate_metrics(y_test, y_pred)
	
	file_path = create_confusion_matrix(name+"_confusion_matrix", y_test, y_pred)

	# Log confusion matrix, model, test metrics, and tags to experiment
	experiment.artifacts.create(key=name+"_confusion_matrix", path=file_path, type="graph")
	
	# Log model to Continual
	experiment.artifacts.create(name+'_model',name+'_model', external=False, upload=True)

	for i in metrics_dict:
		experiment.metrics.create(key=i["key"], value=i["value"], direction=i["direction"], group_name=i["group_name"])

	experiment.tags.create(key="algo", value=name)

def get_metric_id(experiment, key):
    for exp in experiment.metrics.list(page_size=10):
        if exp.key == key:
            return exp.value

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    #parser.add_argument('--alpha', type=float, default=os.environ.get('SM_HP_ALPHA'))
    
    #args, _ = parser.parse_known_args()
    
    CONFIG_KEYS = [
    "CONTINUAL_APIKEY"
    ]
    
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket = 'brendanbucket88',Key = 'dna_sequence_dataset/human.txt')
    dna_data = pd.read_csv(obj['Body'],sep='\t')
    
    config = load_config(CONFIG_KEYS)

    # Create and configure Continual client
    client = Client(api_key=config["CONTINUAL_APIKEY"], endpoint="https://sdk.continual.ai", project="projects/scikit_learn_github_actio_9", environment="production",verify=False)
    run_id = os.environ.get("CONTINUAL_RUN_ID", None)
    run = client.runs.create(description="An example run", run_id=run_id)
    run.state == "ACTIVE"
    
    # Create model
    model = run.models.create("test-on-sagemaker")
    model_version = model.model_versions.create()
    
    # Create dataset object and load data from local text file
    dna_dataset = run.datasets.create("DNA")
    dataset_version = dna_dataset.dataset_versions.create()

    # Profile data
    dataset_version.data_profiles.create(
        dataframes=[dna_data],
        entry_names=["primary_dataset"],
        datetime_columns=["sequence"], # Mush because this dataset doesn't have datetime cols
        index_column="sequence",
        time_index_column="sequence" # Mush because this dataset doesn't have a time index
    )

    # Check data
    #checks = [dict(display_name = "my_data_check", success=True)]
    #dataset_version.create_data_checks(checks)

    # Log dataset 
    artifact_uri = 's3://brendanbucket88/dna_sequence_dataset/human.txt'
    dataset_version.artifacts.create(key = "dna_data", url=artifact_uri, type="csv", external=True)

    X, y = transform(dna_data)
    
    # Loading params
    xgb_params = {
			'max_depth': 3,
			'eta': 0.1,
			'num_class': 3,
			'eval_metric': 'mae',
			'reg_alpha': 0.1 #args.alpha
	}

    # Splitting the human dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)
    xgb=XGBClassifier(xgb_params)
    
    # Create experiment
    xgb_experiment = model_version.experiments.create()
    xgb_experiment.metadata.create(key="training_params", data=xgb_params)
    run_experiment(xgb_experiment, "xgb", xgb, X_train, y_train, y_test)

    # Train second algorithm
    mnb = MultinomialNB(alpha=0.1)
    
    # Create second experiment
    mnb_experiment = model_version.experiments.create()
    run_experiment(mnb_experiment, "mnb", mnb, X_train, y_train, y_test)
