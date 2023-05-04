import re
import pandas as pd
from transformers import pipeline, BertModel


def get_candidates_to_fill_mask(triples):
    hrt_df = pd.DataFrame(triples, columns=['h', 'r', 't'])
    hrt_df['r'] = hrt_df['r'].apply(camel_case_split)
    # head pred and tail pred
    for idx, row in hrt_df.iterrows():
        masked_head_sequence = '[MASK] ' + row['r'] + ' ' + row['t']
        masked_tail_sequence = row['h'] + row['r'] + ' [MASK]'
        head_candidates = fill_masked_node(masked_head_sequence)
        tail_candidates = fill_masked_node(masked_tail_sequence)


def fill_masked_node(sequence):
    bert_model = BertModel.from_pretrained("../saved_models/bert-base-uncased")
    unmasker = pipeline('fill-mask', model=bert_model)
    candidates = unmasker(sequence)
    return [(i['token_str'], i['score']) for i in candidates]


def exists_similar_ents_in_conceptnet(ent):
    pass


def calculate_informative_score(candidate_triples):
    pass


def calculate_domain_and_range_dict():
    hrt_df = pd.DataFrame(data=[], columns=['r', 'h', 't'])
    in_dir1 = "../dataset_only/CN-82K/"
    in_dir2 = "../dataset_only/CN-100K/"
    for in_dir in [in_dir1, in_dir2]:
        for f in ['test', 'valid', 'train']:
            tmp_df = pd.read_csv(in_dir + f"{f}.txt", sep="\t", header=None, names=['r', 'h', 't'])
            tmp_df = tmp_df.apply(lambda x: x.strip())
            hrt_df.append(tmp_df)
    r_df = hrt_df.groupby('r').agg(list)
    r2range = dict()
    r2domain = dict()
    for idx, row in r_df.iterrows():
        r2range.update({row['r']: list(set(row['t']))})
        r2domain.update({row['r']: list(set(row['h']))})
    return r2domain, r2range



def get_hypernym_dict():
    web_is_a = pd.read_csv("../dataset_only/")
    pass


def check_in_range(hrt: list, range_dict):
    # 1. whether in range_dict
    # 2. whether share range hypernym
    return True


def check_in_domain(hrt: list, domain_dict):
    # 1. whether in range_dict
    # 2. whether share range hypernym
    return False


def camel_case_split(x):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', x)



