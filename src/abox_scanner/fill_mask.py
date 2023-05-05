import itertools
import re

import numpy as np
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
    bert_model = BertModel.from_pretrained("../../saved_models/bert-base-uncased")
    unmasker = pipeline('fill-mask', model=bert_model)
    candidates = unmasker(sequence)
    return [(i['token_str'], i['score']) for i in candidates]


def exists_similar_ents_in_conceptnet(ent):
    pass


def calculate_informative_score(candidate_triples):
    pass


def calculate_domain_and_range_dict():
    hrt_df = pd.DataFrame(data=[], columns=['r', 'h', 't'])
    in_dir1 = "../data/conceptnet-82k/"
    in_dir2 = "../data/conceptnet-100k/"
    for in_dir in [in_dir1, in_dir2]:
        for f in ['test', 'valid', 'train']:
            tmp_df = pd.read_csv(in_dir + f"{f}.txt", sep="\t", header=None, names=['r', 'h', 't'])
            tmp_df = tmp_df.applymap(lambda x: x.strip())
            hrt_df = hrt_df.append(tmp_df).drop_duplicates(keep='first')
    r_df = hrt_df.groupby('r', group_keys=True, as_index=False).agg(list)
    r2range = dict()
    r2domain = dict()
    for idx, row in r_df.iterrows():
        r2range.update({'range_' + str(row['r']): list(set(row['t']))})
        r2domain.update({'domain_' + str(row['r']): list(set(row['h']))})
    domain_range = dict()
    domain_range.update(r2range)
    domain_range.update(r2domain)
    sup2sub, disjointness = calculate_class_constraints(domain_range)


def generate_constraints(domain_range, sup2sub, disjointness):
    class2id = {c: i for i, c in enumerate(domain_range.keys())}
    pass




def calculate_class_constraints(r2dict: dict):
    rels = list(r2dict.keys())
    rel_index = range(len(rels))
    overlap_rate = np.zeros((len(rels), len(rels)))
    for i, j in itertools.combinations(rel_index, 2):
        i_rel = rels[i]
        j_rel = rels[j]
        i_instance = r2dict[i_rel]
        j_instance = r2dict[j_rel]
        intersection = list(set(i_instance) & set(j_instance))
        i_in_j = len(intersection) / len(j_instance)
        j_in_i = len(intersection) / len(i_instance)
        overlap_rate[i, j] = i_in_j
        overlap_rate[j, i] = j_in_i
    overlaping_idx = np.argwhere(overlap_rate > 0.5)
    disjointness_idx = np.argwhere(overlap_rate == 0)
    sup_sub = dict()
    disjointness = dict()
    for i, j in overlaping_idx:
        i_rel = rels[i]
        j_rel = rels[j]
        if i_rel in sup_sub and j_rel not in sup_sub[i_rel]:
            sup_sub[i_rel].append(j_rel)
        else:
            sup_sub[i_rel] = [j_rel]
    for i, j in disjointness_idx:
        if i == j:
            continue
        i_rel = rels[i]
        j_rel = rels[j]
        if i_rel in disjointness and j_rel not in disjointness[i_rel]:
            disjointness[i_rel].append(j_rel)
        else:
            disjointness[i_rel] = [j_rel]
    return sup_sub, disjointness


def get_hypernym_dict():
    web_is_a = pd.read_csv("../data/WebisALOD_full.tsv", sep="\t", header=None, names=['ent', 'type', 'score'])
    web_is_a = web_is_a.apply(lambda x: x.strip())
    stopwords = ['thing', 'item', 'factor']
    web_is_a = web_is_a[web_is_a.type.isin(stopwords)==False]
    ent2type = web_is_a[['ent', 'type']].groupby('ent', group_keys=True, as_index=False).agg(list)
    ent2hypernum = {}
    for idx, row in ent2type.iterrows():
        ent2hypernum.update({row['ent']: list(set(row['type']))})
    return ent2hypernum


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


if __name__ == "__main__":
    calculate_domain_and_range_dict()
