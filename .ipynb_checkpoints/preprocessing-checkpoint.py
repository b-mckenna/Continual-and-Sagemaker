import argparse
import os
import boto3
import pandas as pd

CONFIG_KEYS = [
    "CONTINUAL_ENDPOINT",
    "CONTINUAL_APIKEY", 
    "CONTINUAL_PROJECT", 
    "CONTINUAL_ENVIRONMENT"
]

def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

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


def write_transformed_dataset_to_s3:
    ns3 = boto3.resource('s3')
    ns3.Object('brendanbucket88','vectorized_human_data.txt').put(Body=pickle_obj)

if __name__ == "__main__":
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='brendanbucket88',Key='dna_sequence_dataset/human.txt')
    dna_data = pd.read_csv(obj['Body'],sep='\t')
    
    # Create dataset object and load data from local text file
    dna_dataset = run.datasets.create("DNA") 
    dataset_version = dna_dataset.dataset_versions.create()
    
    X, y = transform(dna_data)