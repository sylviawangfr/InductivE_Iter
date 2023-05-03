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
    pass


def get_hypernym(phrase):
    pass


def get_hypernym_dict():
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



