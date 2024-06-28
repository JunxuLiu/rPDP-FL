
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import transformers
from transformers import BertTokenizer
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features
import urllib.request
import warnings
warnings.simplefilter("ignore")
import zipfile

STANFORD_SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
LABEL_LIST = ['contradiction', 'entailment', 'neutral']
NUM_LABELS = 3
MAX_SEQ_LENGHT = 128
NUM_CLIENTS = 10

# Create federated datasets
save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(),"..")), f"iid_{NUM_CLIENTS}")
if not (os.path.exists(save_path)):
    os.mkdir(save_path)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-cased",
    do_lower_case=False,
)

def download_and_extract(dataset_url, data_dir):
    print("Downloading and extracting ...")
    filename = "snli.zip"
    urllib.request.urlretrieve(dataset_url, filename)
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(filename)
    print("Completed!")

def _create_examples(df, set_type):
    """ Convert raw dataframe to a list of InputExample. Filter malformed examples
    """
    examples = []
    for index, row in df.iterrows():
        if row['gold_label'] not in LABEL_LIST:
            continue
        if not isinstance(row['sentence1'], str) or not isinstance(row['sentence2'], str):
            continue
            
        guid = f"{index}-{set_type}"
        if index % 10000 == 0:
            print(guid)
        examples.append(
            InputExample(guid=guid, text_a=row['sentence1'], text_b=row['sentence2'], label=row['gold_label']))
    return examples

def _df_to_features(df, set_type):
    """ Pre-process text. This method will:
    1) tokenize inputs
    2) cut or pad each sequence to MAX_SEQ_LENGHT
    3) convert tokens into ids
    
    The output will contain:
    `input_ids` - padded token ids sequence
    `attention mask` - mask indicating padded tokens
    `token_type_ids` - mask indicating the split between premise and hypothesis
    `label` - label
    """
    examples = _create_examples(df, set_type)
    
    #backward compatibility with older transformers versions
    legacy_kwards = {}
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("2.9.0"):
        legacy_kwards = {
            "pad_on_left": False,
            "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            "pad_token_segment_id": 0,
        }
    
    return glue_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        label_list=LABEL_LIST,
        max_length=MAX_SEQ_LENGHT,
        output_mode="classification",
        **legacy_kwards,
    )

def _features_to_dataset(features):
    """ Convert features from `_df_to_features` into a single dataset
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    return dataset


def iid_partition(dataset, num_clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
    - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
    - clients (int): Number of Clients to split the data between

    returns:
    - Dictionary of image indexes for each client
    """
    num_items_per_client = int(len(dataset)/num_clients)
    items_idxs = list(range(len(dataset)))
    for i in range(num_clients):
        idxs = np.random.choice(items_idxs, num_items_per_client, replace=False)
        items_idxs = list(set(items_idxs) - set(idxs))
        # for num_clients = 10, we have 
        # idxs[:5], len(idxs):
        # [ 34438 414431 475996 430919  83400] 54936
        # [189799 154279 178974  26859 511665] 54936
        # [ 29897 376265 162146 433814 141142] 54936
        # [ 58160  22182 499873 545916 177757] 54936
        # [321549  64629 137271 351410 137850] 54936
        # [375563 388576 231310 410843 543253] 54936
        # [ 90087 526135 164223 139169 156140] 54936
        # [495080 227707 143439  51803  56901] 54936
        # [465157 548450 420936 473619 498130] 54936
        # [493912 353237  17627 267008 199467] 54936

        feature_array = [dataset[idxs][_].numpy() for _ in range(4)]
        combined = list(zip(feature_array[0], feature_array[1], feature_array[2], feature_array[3])) 
        
        cname = 'client{:d}'.format(i) 
        np.save(os.path.join(save_path, f"{cname}.npy"), combined)


dataset_abspath = os.path.abspath(os.path.join(os.getcwd(),"../.."))
download_and_extract(STANFORD_SNLI_URL, dataset_abspath)

data_path = os.path.join(dataset_abspath, "snli_1.0")
train_path =  os.path.join(data_path, "snli_1.0_train.txt")
dev_path = os.path.join(data_path, "snli_1.0_dev.txt")

# read dataframe
df_train = pd.read_csv(train_path, sep='\t')
df_test = pd.read_csv(dev_path, sep='\t')
print("TRAIN Dataset: {}".format(df_train.shape))
print("TEST Dataset: {}".format(df_test.shape))

# dataframe -> TensorDataset
train_features = _df_to_features(df_train, "train")
test_features = _df_to_features(df_test, "test")
train_dataset = _features_to_dataset(train_features)
test_dataset = _features_to_dataset(test_features)

# split into `NUM_CLIENTS`` subsets
print(f"Splitting into {NUM_CLIENTS} subsets ...")
iid_partition(train_dataset, NUM_CLIENTS)