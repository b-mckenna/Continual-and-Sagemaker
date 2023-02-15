from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from continual import Client
from continual.python.sdk.runs import Run
import pickle
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import tempfile
import joblib

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def load_config(CONFIG_KEYS):
    #Loads configurations
    config = {}
    for key in CONFIG_KEYS:
        config[key] = os.getenv(key)
    return config

def kmer(sequence, k=5):
    #Transforms DNA sequence into substrings of length k. 
    return [sequence[x:x+k].lower() for x in range(len(sequence) - k + 1)]

def transform(human_data):
    # Transform dna sequence data into k-mer substrings and vectorize
	human_data['words']=human_data['sequence'].apply(lambda x: kmer(x))
	human_data_revised=human_data.drop(columns='sequence',axis=1)
	human_texts = list(human_data_revised['words'])
	for item in range(len(human_texts)):
		human_texts[item] = ' '.join(human_texts[item])
		
	y=human_data_revised['class'].values
	cv = CountVectorizer(ngram_range=(4,4))
	X = cv.fit_transform(human_texts)
	return X, y

def create_confusion_matrix(name, y_test,y_pred):
    # Create confusion matrix and save it
    file_path = "%s/" %os.getcwd()+name+"-confusion-matrix.png"
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='viridis')
    plt.savefig(file_path)
    return file_path

def run_experiment(experiment, name, model, X_train, y_train, y_test):
    # Fit model
    model.fit(X_train,y_train)

    # Test model
    y_pred=model.predict(X_test)
    
    # Create confusion matrix
    file_path = create_confusion_matrix(name+"-confusion-matrix", y_test, y_pred)
    
    # Log confusion matrix to experiment with Continual
    experiment.artifacts.create(name+"-confusion-matrix", file_path, "graph")
    
    # Log model and metrics to experiment with Continual
    experiment.artifacts.create(name+'-model',name+'-model', external=False, upload=True)
    accuracy = experiment.metrics.create(id="accuracy", display_name="my accuracy")
    precision = experiment.metrics.create(id="precision", display_name="my precision")
    recall = experiment.metrics.create(id="recall", display_name="my recall")
    f1 = experiment.metrics.create(id="f1", display_name="my f1")
    accuracy.log(value=accuracy_score(y_test,y_pred))
    precision.log(value=precision_score(y_test,y_pred,average='weighted'))
    recall.log(value=recall_score(y_test,y_pred,average='weighted'))
    f1.log(value=f1_score(y_test,y_pred,average='weighted'))
    return model
        
def register_winning_experiment(mnb_model, xgb_model, mnb_accuracy, xgb_accuracy, mnb_experiment, xgb_experiment, model_version):
    # Log the model, experiment, and accuracy metric of winning model
    if mnb_accuracy>xgb_accuracy:
        model_version.artifacts.create('mnb-model','mnb-model',upload=True,external=False)
        model_version.metadata.create("best-experiment",mnb_experiment.name)
        model_version.metrics.create(id='accuracy',display_name='accuracy').log(mnb_experiment.metrics.get('accuracy').values[0].value,replace_if_exists=True)
        return mnb_model
    elif mnb_accuracy<xgb_accuracy:
        model_version.artifacts.create('xgb-model','xgb-model',upload=True,external=False)
        model_version.metadata.create('best-experiment',xgb_experiment.name)
        model_version.metrics.create('accuracy','accuracy').log(xgb_experiment.metrics.get('accuracy').values[0].value,replace_if_exists=True)
        return xgb_model
    else:
        model_version.artifacts.create('mnb-model','mnb-model',upload=True,external=False)
        return mnb_model

if __name__ == "__main__":
    
    CONFIG_KEYS = [
    "CONTINUAL_API_KEY"
    ]
    
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket = 'brendanbucket88',Key = 'dna_sequence_dataset/human.txt')
    dna_data = pd.read_csv(obj['Body'],sep='\t')
    
    config = load_config(CONFIG_KEYS)

    # Create and configure Continual client
    client = Client(api_key=config["CONTINUAL_API_KEY"], endpoint="https://sdk.continual.ai", project="projects/continual-and-sagemaker", environment="production",verify=False)
    run_id = os.environ.get("CONTINUAL_RUN_ID", None)
    run = client.runs.create(description="An example run", id=run_id)
    
    # Create model
    model = run.models.create("Gene-Classification",replace_if_exists=True)
    model_version = model.model_versions.create(replace_if_exists=True)

    # Log dataset 
    dataset = run.datasets.create(display_name="HumanDNA", description="DNA sequencing dataset")
    dataset_version = dataset.dataset_versions.create()
    artifact_uri = 's3://brendanbucket88/dna_sequence_dataset/human.txt'
    dataset_version.artifacts.create("dna-data", url=artifact_uri, external=True)

    X, y = transform(dna_data)
    
    # Loading params
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--alpha', type=float, default=os.environ.get('SM_HP_ALPHA'))
    parser.add_argument('--max_depth', type=int, default=os.environ.get('SM_HP_MAX_DEPTH'))
    parser.add_argument('--eta', type=float, default=os.environ.get('SM_HP_ETA'))
    parser.add_argument('--num_class', type=int, default=os.environ.get('SM_HP_NUM_CLASS'))
    parser.add_argument('--eval_metric', type=str, default=os.environ.get('SM_HP_EVAL_METRIC'))

    args, _ = parser.parse_known_args()
    
    xgb_params = {
			'max_depth': args.max_depth,
			'eta': args.eta,
			'num_class': args.num_class,
			'eval_metric': args.eval_metric,
			'reg_alpha': args.alpha
	}

    # Split the human dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)
    
    # Create first algorithm
    xgb=XGBClassifier(xgb_params)
    
    # Create first experiment
    xgb_experiment = model_version.experiments.create()
    xgb_experiment.metadata.create("training-params", data=xgb_params)
    xgb_model = run_experiment(xgb_experiment, "xgb", xgb, X_train, y_train, y_test)

    # Train second algorithm
    mnb = MultinomialNB(alpha=0.1)
    
    # Create second experiment
    mnb_experiment = model_version.experiments.create()
    mnb_model = run_experiment(mnb_experiment, "mnb", mnb, X_train, y_train, y_test)
    
    # Retrieve accuracy metrics from Continual
    xgb_accuracy = xgb_experiment.metrics.get("accuracy").values[0].value
    mnb_accuracy = mnb_experiment.metrics.get("accuracy").values[0].value
    
    final_model = register_winning_experiment(mnb_model, xgb_model, mnb_accuracy, xgb_accuracy, mnb_experiment, xgb_experiment, model_version)
    
    # Save model for Sagemaker
    joblib.dump(final_model, "/opt/ml/model/model.joblib")
    
    # Save model to custom S3 bucket
    loc = 'models/'
    model_filename = '.sav'
    rid = model_version.run.split('/')[5]
    OutputFile = loc + str(rid) + model_filename

    # Write model locally then write it to S3 bucket. 
    with tempfile.TemporaryFile() as fp:
        joblib.dump(final_model, fp)
        s3.put_object(Bucket='brendanbucket88', Key=OutputFile, Body=fp.read())
