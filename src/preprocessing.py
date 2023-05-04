import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def generate_bert_feature(args):
    entities = read_entities()
    bert_path = "../saved_models/bert-base-uncased"
    local_models = Path(bert_path)
    if local_models.exists():
        bert_encoder = BertModel.from_pretrained(bert_path)
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        # model = BertModel.from_pretrained('bert-base-uncased')
        ent2feature = dict()
        for ent in tqdm(entities):
            input_ids = torch.tensor(tokenizer.encode(ent)).unsqueeze(0)  # Batch size 1
            outputs = bert_encoder(input_ids)
            last_hidden_states = outputs[0].mean(1).squeeze()  # The last hidden-state is the first element of the output tuple
            ent2feature.update({ent: last_hidden_states.detach().cpu()})
        with open(args.out_dir + 'ent2bert.pkl', 'wb') as f:
            # write the python object (dict) to pickle file
            pickle.dump(ent2feature, f)


def read_entities():
    entities = []
    in_dir1 = "../data/conceptnet-82k/"
    in_dir2 = "../data/conceptnet-100k/"
    for in_dir in [in_dir1, in_dir2]:
        for f in ['train', 'test', 'valid']:
            df = pd.read_csv(in_dir + f"{f}.txt", sep="\t", header=None, names=['r', 'h', 't'])
            df = pd.concat([df['r'], df['h'], df['t']]).apply(lambda x: x.strip()).unique()
            entities.extend(df.tolist())
            entities = list(set(entities))
    return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument('--out_dir', type=str, default="../data/saved_entity_embedding/conceptnet/")
    args = parser.parse_args()
    generate_bert_feature(args)