import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import json

#logging.basicConfig(filename='output.log', level=logging.INFO)

#######################################################################
# Utility functions for producing
#######################################################################
from src import utils, reader_utils, data_loader
from src.graph import Graph
from src.model import LinkPredictor
from src.reader import Reader


def pred_head(model, test_triplets, e1_to_multi_e2, network, num_rels):
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
        if target == 'tail':
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


class ConceptNetProducerReader(Reader):
    def __init__(self):
        self.graph = Graph()
        self.rel2id = {}

    def read_network(self, hrt_df: pd.DataFrame, train_network=None):
        acc_add_nodes = 0
        for idx, row in hrt_df.iterrows():
            self.add_example(row['h'], row['t'], row['r'], float(0))
            _, new_added = self.add_example(row['h'], row['t'], row['r'], float(1), int(1), train_network)
            acc_add_nodes += new_added
        self.rel2id = self.graph.relation2id

    def add_example(self, src, tgt, relation, weight, label=1, train_network=None):
        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)
        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)
        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)
        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                   self.graph.nodes[tgt_id],
                                   self.graph.relations[relation_id],
                                   label,
                                   weight)
        # add nodes/relations from evaluation graphs to training graph too
        new_added = 0
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)
                new_added += 1
            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)
                new_added += 1
            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)
        return edge, new_added


def load_test(hrt_df, train_network):
    # load graph data
    test_network = ConceptNetProducerReader()
    test_network.read_network(hrt_df, train_network=train_network)
    word_vocab = train_network.graph.node2id
    test_data, _ = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)
    test_data = torch.LongTensor(test_data)
    return test_data


def produce(args):
    # set random seed
    # utils.set_seeds(args.seed)
    # check cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # - - - - - - - - - - - - - - - - - Data Loading - - - - - - - - - - - - - - - - - - -
    # Load train data only
    train_data, valid_data, test_data, train_network, id2node = data_loader.load_data(args)
    num_nodes = len(train_network.graph.nodes)
    num_rels = len(train_network.graph.relations)
    args.num_nodes = num_nodes
    args.num_rels = num_rels
    # Load test data


    # calculate degrees for entities
    _, degrees, _, _ = utils.get_adj_and_degrees(num_nodes, num_rels, train_data)
    # - - - - - - - - - - - - - - - - - Statistics for evaluation - - - - - - - - - - - - - - - - - - -
    all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()
    # for filtered ranking (for evaluation)
    all_e1_to_multi_e2, all_e2_to_multi_e1 = reader_utils.create_entity_dicts(all_tuples, num_rels)

    # - - - - - - - - - - - - - - - - - Pre-trained feature loading - - - - - - - - - - - - - - - - - - -
    # Embedding initialization
    if args.bert_feat_path != 'None' and args.fasttext_feat_path != 'None':
        bert_feature = utils.load_pre_computed_feat(args.bert_feat_path, args.bert_feat_dim, id2node)
        fasttext_feature = utils.load_pre_computed_feat(args.fasttext_feat_path, args.fasttext_feat_dim, id2node)
        fusion_feature = torch.cat((bert_feature, fasttext_feature),dim=1)
        print("Loading Pre-computed BERT and fasttext Embedding")
    elif args.bert_feat_path != 'None':
        bert_feature = utils.load_pre_computed_feat(args.bert_feat_path, args.bert_feat_dim, id2node)
        print("Loading Pre-computed BERT Embedding")
    elif args.fasttext_feat_path != 'None':
        fasttext_feature = utils.load_pre_computed_feat(args.fasttext_feat_path, args.fasttext_feat_dim, id2node)
        print("Loading Pre-computed fasttext Embedding")
    else:
        print("No node feature provided. Use random initialization")
    print('')

    # - - - - - - - - - - - - - - - - - Fixed Graph Preparation - - - - - - - - - - - - - - - - - - -
    # Fixed graph
    fix_edge_src = []
    fix_edge_tgt = []
    fix_edge_type = []

    args.num_edge_types = 0

    # create a triplet graph (with edge types)
    tri_edge_src, tri_edge_tgt, tri_edge_type = utils.create_triplet_graph(args, train_data.tolist())
    tri_graph = (tri_edge_src, tri_edge_tgt, tri_edge_type)
    print('Number of triplet edges: ', len(tri_edge_src))
    print('Number of triplet edges types: ', args.num_rels * 2)
    print('')
    fix_edge_src.extend(tri_graph[0])
    fix_edge_tgt.extend(tri_graph[1])
    fix_edge_type.extend(tri_graph[2])
    args.num_edge_types = args.num_edge_types + args.num_rels * 2

    # Add similarity graph edge
    print('Add sim edge type for semantic similarity graph')
    args.num_edge_types = args.num_edge_types + 1

    print('Number of relation types for R-GCN model: ', args.num_edge_types)
    fixed_graph = (fix_edge_src, fix_edge_tgt, fix_edge_type)
    print('Total number of fixed edges: ', len(fix_edge_src))
    print('')
    # create model
    model = LinkPredictor(args)
    print(model)
    # embedding initialization
    if args.bert_feat_path != 'None' and args.fasttext_feat_path != 'None':
        print("Initialize with concatenated BERT and fastText Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': fusion_feature})
    elif args.bert_feat_path != 'None':
        print("Initialize with Pre-computed BERT Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': bert_feature})
    elif args.fasttext_feat_path != 'None':
        print("Initialize with Pre-computed fasttext Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': fasttext_feature})
    else:
        print("No node feature provided. Use uniform initialization")
    # model.to(args.device)

    # - - - - - - - - - - - - - - Evaluation Only - - - - - - - - - - - - - - - - - - -
    # TODO, not finished
    if args.load_model:
        model_state_file = args.load_model
    else:
        print("Please provide model path for evaluation (--load_model)")
        sys.exit(0)

    checkpoint = torch.load(model_state_file)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])

    eval_graph = checkpoint['eval_graph']
    print("Using best epoch: {}".format(checkpoint['epoch']))

    # Update whole graph embedding
    g_whole, node_id, node_norm = utils.sample_sub_graph(args, 1000000000, eval_graph)
    if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
        print('perform evaluation on cpu')
        model.cpu()
        g_whole = g_whole.cpu()
        g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
        g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
        g_whole.edata['type'] = g_whole.edata['type'].cpu()

    # update all embedding
    if model.entity_embedding != None:
        del model.entity_embedding
        model.entity_embedding = None
        torch.cuda.empty_cache()
    node_id_copy = np.copy(node_id)
    model.update_whole_embedding_matrix(g_whole, node_id_copy)

    # evaluation_utils.ranking_and_hits(args, model, valid_data, all_e1_to_multi_e2, train_network)
    print("================Produce Head=================")
    produce_head_data = load_test(args.pred_head_df, train_network)
    produce_tail_data = load_test(args.pred_tail_df, train_network)
    pred_head(model, produce_head_data, all_e1_to_multi_e2, train_network, args.num_rels)
    pred_tail(model, produce_tail_data, all_e1_to_multi_e2, train_network)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--dataset', type=str, default="conceptnet-82k")
    parser.add_argument('--load_model', type=str, default="../saved_ckg_model/May_10_10_09_41.pt")
    # parser.add_argument('--evaluate_every', type=int, default=15)
    parser.add_argument('--output_dir', type=str, default="../saved_ckg_model/")
    parser.add_argument('--bert_feat_path', type=str, default="../data/saved_entity_embedding/conceptnet/ent2bert.pkl")
    parser.add_argument('--decoder_embedding_dim', type=int, default=500)
    parser.add_argument('--decoder_batch_size', type=int, default=256)
    # parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--decoder', type=str, default="ConvTransE")
    # parser.add_argument('--patient', type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument('--regularization', type=float, default=1e-25)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--input_dropout', type=float, default=0.15)
    parser.add_argument("--feature_map_dropout", type=float, default=0.15)
    # parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--bert_feat_dim", type=int, default=768)
    # parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--dec_kernel_size", type=int, default=5)
    parser.add_argument("--dec_channels", type=int, default=300)
    parser.add_argument("--encoder", type=str, default="RWGCN_NET")
    parser.add_argument("--graph_batch_size", type=int, default=50000)
    parser.add_argument("--entity_feat_dim", type=int, default=768)
    parser.add_argument("--fasttext_feat_path", type=str, default='None')
    parser.add_argument("--fasttext_feat_dim", type=int, default=300)
    parser.add_argument("--gnn_dropout", type=float, default=0.2)
    parser.add_argument("--n_ontology", type=int, default=5)
    # parser.add_argument("--dynamic_graph_ee_epochs", type=int, default=100)
    # parser.add_argument("--start_dynamic_graph", type=int, default=50)
    parser.add_argument("--rel_regularization", type=float, default=0.1)
    # parser.add_argument("--fix_triplet_graph", type=bool, default=True)
    # parser.add_argument("--dynamic_sim_graph", type=bool, default=True)
    # parser.add_argument("--eval_only", type=bool, default=False)
    # parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--label_smoothing_epsilon", type=float, default=0.0001)
    # parser.add_argument("--clean_update", type=int, default=2)
    parser.add_argument("--grad_norm", type=float, default=0.0001)
    parser.add_argument("--num_hidden", type=int, default=2)
    parser.add_argument("--l_relu_ratio", type=float, default=0.001)
    # Parsing all hyperparameters
    args = parser.parse_args()
    test = pd.read_csv("../data/conceptnet-82k/test.txt", sep="\t", header=None, names=['r', 'h', 't'])
    args.pred_head_df = test.sample(5)
    args.pred_tail_df = test.sample(5)
    # Run main function
    try:
        produce(args)
    except KeyboardInterrupt:
        print('Interrupted')