import argparse
from codecs import ignore_errors
from random import shuffle
import time
import os, sys, shutil
from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
import networkx as nx
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
from easylink.common.eval_utils import evaluate_auc
from easylink.model.heuristic_similarity import common_neighbors, adamic_adar, resource_allocation, local_path_index
from easylink.model.seal import SEAL, SEALDataset
from easylink.common.data_utils import load_basic_network, train_test_split

@torch.no_grad()
def test(dataset, seal, batch_size, evaluator=None):
    seal.model.eval()
    data_loader = DataLoader(dataset, batch_size)
    pbar = tqdm(data_loader, ncols=80)
    y_pred, y_true = [], []
    for data in pbar:
        data = data.to(device)
        x = data.x if args.use_feature else None
        logits = seal.model(data.z, data.edge_index, data.batch, x)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    result = {}
    auc = roc_auc_score(val_true, val_pred)
    print("AUC:{}".format(auc))
    result['AUC'] = auc

    if evaluator is not None:
        for K in [20, 50, 100]:
            evaluator.K = K
            valid_hits = evaluator.eval({
                    'y_pred_pos': pos_val_pred,
                    'y_pred_neg': neg_val_pred,
                })[f'hits@{K}']
            result[f'Hits@{K}'] = valid_hits
            print(f"Hits@{K}:{valid_hits}")

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEAL LinkPredictor')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--use_cache', action='store_true')
    # DataStructure settings
    parser.add_argument('--use_feature', action='store_true', 
                    help="whether to use raw node features as GNN input")
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--max_nodes_per_hop', type=int, default=10)
    # GNN settings
    parser.add_argument('--model', type=str, default='SAGE')
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)

    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--debug_samples', type=int, default=0)
    args = parser.parse_args()
    print(args)
    
    # Data Preparation
    if args.dataset.startswith('ogbl'):
        print("Loading data.")
        dataset_name = args.dataset
        dataset_root = '../data'
        dataset = PygLinkPropPredDataset(
            name=dataset_name, root=dataset_root)
        evaluator = Evaluator(dataset_name)
        split_edge = dataset.get_edge_split()
        graph = dataset[0]
        train_pos_edges = split_edge['train']['edge']
        if args.debug_samples > 0:
            train_pos_edges = train_pos_edges[:args.debug_samples]
        val_pos_edge, val_neg_edge = split_edge['valid']['edge'], split_edge['valid']['edge_neg']
        test_pos_edge, test_neg_edge = split_edge['test']['edge'], split_edge['test']['edge_neg']
        
        # Training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        max_z = 10000
        node_feat = graph.x if args.use_feature else None
        feat_dim = dataset.num_node_features if args.use_feature else args.hidden_channels
        print(feat_dim)
        base_edges = graph.edge_index.t()
        seal = SEAL('SAGE', args.use_feature, args.lr, feat_dim, args.hidden_channels, args.num_layers, max_z, args.dropout)
        train_dataset_dir = dataset.root+'_seal'
        if not args.use_cache:
            shutil.rmtree(train_dataset_dir, ignore_errors=True)
        train_dataset = SEALDataset(train_dataset_dir, base_edges, train_pos_edges, 
                        graph.num_nodes, args.num_hops, args.max_nodes_per_hop, node_feat=node_feat)
        seal.train(train_dataset, args.epochs, args.batch_size, device)

        # Valid
        val_dataset_dir = dataset.root+"_seal_val"
        if not args.use_cache:
            shutil.rmtree(val_dataset_dir, ignore_errors=True)
        val_dataset = SEALDataset(val_dataset_dir, base_edges, val_pos_edge,
                                graph.num_nodes, args.num_hops, args.max_nodes_per_hop,
                                node_feat=node_feat, neg_edges=val_neg_edge)
        print("Validation:")
        test(val_dataset, seal, args.batch_size, evaluator)

        # Test
        test_dataset_dir = dataset.root+"_seal_test"
        if not args.use_cache:
            shutil.rmtree(test_dataset_dir, ignore_errors=True)
        test_dataset = SEALDataset(test_dataset_dir, base_edges, test_pos_edge,
                                graph.num_nodes, args.num_hops, args.max_nodes_per_hop,
                                node_feat=node_feat, neg_edges=test_neg_edge)
        print("Test:")
        test(test_dataset, seal, args.batch_size, evaluator)
    else:
        dataset_root = '../data/facebook'
        # facebook
        facebook_dir = dataset_root+'/facebook.txt'
        g = load_basic_network(facebook_dir)
        adj = nx.adjacency_matrix(g)
        train_edges, test_edges, val_edges, train_neg_edges, test_neg_edges, val_neg_edges = train_test_split(adj)

        test_heuristics = False
        if test_heuristics:
            train_edges = np.array(train_edges)
            train_edges_reverse = np.array([train_edges[:,1], train_edges[:,0]]).transpose()
            train_edges = np.concatenate([train_edges, train_edges_reverse], axis=0)
            edge_weight = torch.ones(train_edges.shape[0], dtype=int)
            A = ssp.csr_matrix(
                (edge_weight, (train_edges[:,0], train_edges[:,1])), shape = (g.number_of_nodes(), g.number_of_nodes())
            )
            h_predictor = common_neighbors 
            val_edges = torch.tensor(val_edges)
            val_neg_edges = torch.tensor(val_neg_edges)
            pos_valid_pred = h_predictor(A, val_edges, batch_size=args.batch_size)
            neg_valid_pred = h_predictor(A, val_neg_edges, batch_size=args.batch_size)
            val_pred = torch.cat([torch.Tensor(pos_valid_pred), torch.Tensor(neg_valid_pred)])
            val_true = torch.cat([torch.ones(pos_valid_pred.shape[0], dtype=int),
                        torch.zeros(neg_valid_pred.shape[0], dtype=int)]) 
            auc = roc_auc_score(val_true, val_pred)
            print("Heuristic AUC:{}".format(auc)) 
        else:
            train_edges = np.array(train_edges)
            train_edges_reverse = np.array([train_edges[:,1], train_edges[:,0]]).transpose()
            train_edges = np.concatenate([train_edges, train_edges_reverse], axis=0)
            train_edges = torch.tensor(train_edges)
            val_edges = torch.tensor(val_edges)
            val_neg_edges = torch.tensor(val_neg_edges)
            # Training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            max_z = 1000
            seal = SEAL('SAGE', False, args.lr, 0, args.hidden_channels, args.num_layers, max_z, args.dropout)
            train_dataset_dir = dataset_root+'_seal'
            if not args.use_cache:
                shutil.rmtree(train_dataset_dir, ignore_errors=True)
            train_dataset = SEALDataset(train_dataset_dir, train_edges, train_edges[:100000], 
                            g.number_of_nodes(), args.num_hops, args.max_nodes_per_hop, node_feat=None)
            seal.train(train_dataset, args.epochs, args.batch_size, device)

            # Test
            val_dataset_dir = dataset_root+"_seal_val"
            if not args.use_cache:
                shutil.rmtree(val_dataset_dir, ignore_errors=True)
            val_dataset = SEALDataset(val_dataset_dir, train_edges, val_edges,
                                    g.number_of_nodes(), args.num_hops, args.max_nodes_per_hop,
                                    node_feat=None, neg_edges=val_neg_edges)
            test(val_dataset, seal, args.batch_size, None)
