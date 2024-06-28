import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datasets import load_dataset


def preprocess(data_split):
    #https://huggingface.co/datasets/stanfordnlp/snli
    dataset = load_dataset("stanfordnlp/snli", name="plain_text", split=data_split)
    df = dataset.to_pandas()
    df = remove_extra_label(df)
    premises = df['premise'].tolist()
    hypotheses = df['hypothesis'].tolist()
    labels = df['label'].tolist()
    labels = torch.tensor(labels)
    text = [list(item) for item in zip(premises, hypotheses)]
    labels = df['label'].tolist()
    return text,labels

def remove_extra_label(df):
    #print(df['label'].unique())
    num_unique_values=len(df['label'].unique())
    value_counts = df['label'].value_counts()

    # Remove rows where 'label' is -1
    df = df[df['label'] != -1]
    data_files = df[df['label'] != -1]
    df = df[df['label'] != -1]

    #make sure we have only 3 unique values
    #print(df['label'].unique())
    num_unique_values = len(df['label'].unique())
    value_counts = df['label'].value_counts()
    return df


#usage
'''
text_train,labels_train=preprocess(data_split="train")
text_validation,labels_validation=preprocess(data_split="validation")
text_test,label_test=preprocess(data_split="test")
'''