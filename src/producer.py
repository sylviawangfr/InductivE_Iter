import numpy as np
import torch
import json

#logging.basicConfig(filename='output.log', level=logging.INFO)

#######################################################################
# Utility functions for producing
#######################################################################


def prd_head(model, test_triplets, e1_to_multi_e2, network, num_rels):
    s = test_triplets[:, 0]
    r = test_triplets[:, 1]
    o = test_triplets[:, 2]
    rel_offset = num_rels
    scores = []
    batch_size = 64
    end = len(test_triplets)

    for i in range(0, end, batch_size):
        e1 = s[i: i + batch_size]
        e2 = o[i: i + batch_size]
        rel = r[i: i + batch_size]
        rel_reverse = rel + rel_offset
        cur_batch_size = len(e1)
        e2_multi2 = [torch.LongTensor(e1_to_multi_e2[(e.cpu().item(), r.cpu().item())]) for e, r in
                     zip(e2, rel_reverse)]

        with torch.no_grad():
            pred2 = model(e2, rel_reverse)
        e1, e2 = e1.data, e2.data
        for j in range(0, cur_batch_size):
            # these filters contain ALL labels
            filter2 = e2_multi2[j].long()
            # save the prediction that is relevant
            # target_value2 = pred2[j, e1[j].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred2[j][filter2] = 0.0
            # EXP: also remove self-connections
            pred2[j][e2[j].item()] = 0.0
            # write base the saved values
            # pred2[j][e1[j]] = target_value2
        pred2 = pred2.data.cpu()
        scores.append(pred2)
    candidates = get_topk_tuples(torch.cat(scores, dim=0).cpu().numpy(), test_triplets, network, target='head')
    with open("topk_head_candidates.jsonl", 'w') as f:
        for entry in candidates:
            json.dump(entry, f)
            f.write("\n")


def pred_tail(model, test_triplets, e1_to_multi_e2, network):
    s = test_triplets[:, 0]
    r = test_triplets[:, 1]
    o = test_triplets[:, 2]
    hits_right = []
    hits = []
    scores = []
    for i in range(10):
        hits_right.append([])
        hits.append([])
    batch_size = 64
    end = len(test_triplets)
    
    for i in range(0, end, batch_size):
        e1 = s[i: i + batch_size]
        e2 = o[i: i + batch_size]
        rel = r[i: i + batch_size]
        cur_batch_size = len(e1)
        e2_multi1 = [torch.LongTensor(e1_to_multi_e2[(e.cpu().item(), r.cpu().item())]) for e, r in zip(e1, rel)]
        with torch.no_grad():
            pred1 = model(e1, rel)
        e1, e2 = e1.data, e2.data
        for j in range(0, cur_batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[j].long()
            # save the prediction that is relevant
            target_value1 = pred1[j, e2[j].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[j][filter1] = 0.0
            # EXP: also remove self-connections
            pred1[j][e1[j].item()] = 0.0
            # write base the saved values
            # pred1[j][e2[j]] = target_value1
        pred1 = pred1.data.cpu()
        scores.append(pred1)
    candidates = get_topk_tuples(torch.cat(scores, dim=0).cpu().numpy(), test_triplets, network, target='tail')
    with open("topk_tail_candidates.jsonl", 'w') as f:
        for entry in candidates:
            json.dump(entry, f)
            f.write("\n")


def get_topk_tuples(scores, input_prefs, network, target, k=5):
    out_lines = []
    argsort = [np.argsort(-1 * np.array(score)) for score in np.array(scores)]

    for i, sorted_scores in enumerate(argsort):
        pref = input_prefs[i]
        e1 = pref[0].cpu().item()
        rel = pref[1].cpu().item()
        e2 = pref[2].cpu().item()
        cur_point = {'gold_triple': {}}
        cur_point['gold_triple']['e1'] = network.graph.nodes[e1].name
        cur_point['gold_triple']['e2'] = network.graph.nodes[e2].name
        cur_point['gold_triple']['relation'] = network.graph.relations[rel].name

        topk_indices = sorted_scores[:k]
        topk_tuples = [network.graph.nodes[elem] for elem in topk_indices]
        cur_point['candidates'] = []
        if target is 'tail':
            for j, node in enumerate(topk_tuples):
                tup = {'e1': network.graph.nodes[e1].name, 'e2': node.name,
                       'relation': network.graph.relations[rel].name, 'score': str(scores[i][topk_indices[j]])}
                cur_point['candidates'].append(tup)
        else:
            for j, node in enumerate(topk_tuples):
                tup = {'e1': node.name, 'e2': network.graph.nodes[e2].name,
                       'relation': network.graph.relations[rel].name, 'score': str(scores[i][topk_indices[j]])}
                cur_point['candidates'].append(tup)
        out_lines.append(cur_point)
    return out_lines
