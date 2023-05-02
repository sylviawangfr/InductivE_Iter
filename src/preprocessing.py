import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def generate_bert_feature(args):
    entities = read_entities(args.in_dir)
    bert_path = "../saved_models/bert-base-uncased"
    local_models = Path(bert_path)
    if local_models.exists():
        bert_encoder = BertModel.from_pretrained(bert_path)
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        # model = BertModel.from_pretrained('bert-base-uncased')
        ent2feature = dict()
        i = 0
        for ent in tqdm(entities):
            input_ids = torch.tensor(tokenizer.encode(ent)).unsqueeze(0)  # Batch size 1
            outputs = bert_encoder(input_ids)
            last_hidden_states = outputs[0].mean(1)  # The last hidden-state is the first element of the output tuple
            ent2feature.update({ent: last_hidden_states})
            i += 1
            if i == 100:
                break
        with open(args.out_dir + 'ent2bert.pkl', 'wb') as f:
            # write the python object (dict) to pickle file
            pickle.dump(ent2feature, f)


def read_entities(in_dir):
    entities = []
    for f in ['test', 'valid', 'train']:
        df = pd.read_csv(in_dir + f"{f}.txt", sep="\t", header=None)
        df = pd.concat([df[0], df[1], df[2]]).unique()
        entities.extend(df.tolist())
    entities = list(set(entities))
    return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument('--in_dir', type=str, default="../data/conceptnet-82k/")
    parser.add_argument('--out_dir', type=str, default="../data/saved_entity_embedding/conceptnet/")
    args = parser.parse_args()
    generate_bert_feature(args)