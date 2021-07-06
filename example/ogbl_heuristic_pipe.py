import os
import sys
import argparse

import networkx as nx
from networkx.classes.function import common_neighbors
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm


parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
from easylink.model.heuristic_similarity import common_neighbors, adamic_adar, resource_allocation, local_path_index
from easylink.common.eval_utils import evaluate_hits, evaluate_auc

# sample run: python ogbl_heuristic_pipe.py --dataset ogbl-collab --model adamic_adar --test 1 

def main():
    # Data settings
    parser = argparse.ArgumentParser(description='OGBL (SEAL)')
    parser.add_argument('--dataset', type=str, default='ogbl-collab', help='choose: ogbl-ppa, ogbl_collab, ogbl-ddi')
    parser.add_argument('--model', type=str, default='adamic_adar', help="choose: common_neighbors, adamic_adar, resource_allocation, local_path_index")   
    parser.add_argument('--test', type=int, default=1, help="sample small number of data to run pipeline.")    

    args = parser.parse_args()

    print("Loading data.")
    dataset_name = args.dataset
    dataset = PygLinkPropPredDataset(name=dataset_name, root='../data/')
    evaluator = Evaluator(name=dataset_name)
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # graph_construct
    print("Constructing graph.")
    train_edges_raw = np.array(split_edge['train']['edge'])
    train_edges_reverse = np.array([train_edges_raw[:,1], train_edges_raw[:,0]]).transpose()
    train_edges = np.concatenate([train_edges_raw, train_edges_reverse], axis=0)
    edge_weight = torch.ones(train_edges.shape[0], dtype=int)
    A = ssp.csr_matrix(
        (edge_weight, (train_edges[:,0], train_edges[:,1])), shape = (data.num_nodes, data.num_nodes)
    )

    # test 
    print("Benchmark test.")
    batch_size = 1024
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']

    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    if args.test == 1:
        pos_valid_edge = pos_valid_edge[:1000]
        neg_valid_edge = neg_valid_edge[:1000]
        pos_test_edge = pos_test_edge[:1000]
        neg_test_edge = neg_test_edge[:1000]
        
    model_predictor = eval(args.model) # use model_name as function
    print("Model: {}".format(args.model))
    pos_valid_pred = model_predictor(A, pos_valid_edge, batch_size=batch_size)
    neg_valid_pred = model_predictor(A, neg_valid_edge, batch_size=batch_size)
    
    if dataset_name == 'ogbl_collab':
        # adding valid data when predict testing
        train_valid_edges_raw = np.concatenate((np.array(split_edge['train']['edge']),np.array(split_edge['valid']['edge'])))
        train_valid_edges_reverse = np.array([train_valid_edges_raw[:,1], train_valid_edges_raw[:,0]]).transpose()
        train_valid_edges = np.concatenate([train_valid_edges_raw, train_valid_edges_reverse], axis=0)
        train_valid_edge_weight = torch.ones(train_valid_edges.shape[0], dtype=int)
        A2 = ssp.csr_matrix(
            (train_valid_edge_weight, (train_valid_edges[:,0], train_valid_edges[:,1])), shape = (data.num_nodes, data.num_nodes)
        )
        pos_test_pred = model_predictor(A2, pos_test_edge)
        neg_test_pred = model_predictor(A2, neg_test_edge)
    else:
        pos_test_pred = model_predictor(A, pos_test_edge)
        neg_test_pred = model_predictor(A, neg_test_edge)
    
    eval_res = evaluate_hits(evaluator, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    for key, result in eval_res.items():
        valid_hits, test_hits = result
        print(key)
        print(
            f'Valid: {100 * valid_hits:.2f}%, '
            f'Test: {100 * test_hits:.2f}%')

    val_pred = torch.cat([torch.Tensor(pos_valid_pred), torch.Tensor(neg_valid_pred)])
    val_true = torch.cat([torch.ones(pos_valid_pred.shape[0], dtype=int),
                        torch.zeros(neg_valid_pred.shape[0], dtype=int)])
    test_pred = torch.cat([torch.Tensor(pos_test_pred), torch.Tensor(neg_test_pred)])
    test_true = torch.cat([torch.ones(pos_test_pred.shape[0], dtype=int),
                        torch.zeros(neg_test_pred.shape[0], dtype=int)])
    auc_results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    print("Valid AUC, Test AUC")
    print(auc_results)

if __name__ == '__main__':
    main()