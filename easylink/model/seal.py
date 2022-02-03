import argparse
from random import shuffle
import time
import os, sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.nn import (BCEWithLogitsLoss)
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
parent_path = os.path.dirname(os.path.dirname(sys.path[0]))
print(parent_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)
from easylink.common.seal_utils import *

class SEAL():
    '''
        Link Prediction based on Subgraph, Attributes and Embedding.
        cite: [Link prediction based on graph neural networks](https://arxiv.org/abs/1802.09691)

        SEAL is a GNN-based link prediction method. It first extracts a k-hop enclosing subgraph 
        for each target link, then applies a labeling trick named Double Radius Node Labeling (DRNL) 
        to give each node an integer label as its additional feature. 
        Finally, these labeled enclosing subgraphs are fed to a graph neural network to predict 
        link existences.

        Code partially refer to [SEAL_OGB](https://github.com/facebookresearch/SEAL_OGB)

    '''
    def __init__(self, gnn, args, max_z=10000):
        '''
            args:
            - gnn: the name of gnn model
            - args: config parameters
            - max_z: num of embeddings each sub-graph
        '''
        self.model = None
        self.args = args
        if gnn == 'SAGE':
            self.model = SAGE(args.hidden_channels, args.num_layers, max_z, args.dropout)
        parameters = list(self.model.parameters())
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')
        self.optimizer = torch.optim.Adam(params=parameters, lr=args.lr)

    def train(self, train_dataset, device, args):
        self.model.train()
        total_loss, cnt = 0., 0
        for epoch in range(args.epochs):
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            pbar = tqdm(train_loader, ncols=80)
            for data in pbar:
                data = data.to(device)
                self.optimizer.zero_grad()
                x = data.x if self.args.use_feature else None
                logits = self.model(data.z, data.edge_index, data.batch, x)
                loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                cnt += len(logits)
                cur_loss = round(total_loss/cnt,6)
                pbar.set_description("Epoch {}, Loss: {}".format(epoch, cur_loss))

        return total_loss / cnt

    @torch.no_grad()
    def predict(self, dataset, batch_size, device):
        self.model.eval()
        data_loader = DataLoader(dataset, batch_size)
        pbar = tqdm(data_loader, ncols=80)
        y_pred = []
        for data in pbar:
            data = data.to(device)
            x = data.x if self.args.use_feature else None
            logits = self.model(data.z, data.edge_index, data.batch, x)
            y_pred.append(logits.view(-1).cpu())
        return torch.cat(y_pred)


class SEALDataset(InMemoryDataset):
    '''
        Construct each pair of link into subgraph based on labeling trick
    '''
    def __init__(self, root, edge_index, pos_edges, num_nodes, num_hops, max_nodes_per_hop=10, 
                 node_feat=None, neg_edges=None, node_label='drnl', dataset_name='default'):
        self.edge_index = edge_index
        self.pos_edge_index = pos_edges.t() if pos_edges is not None else None
        self.num_nodes = num_nodes
        self.num_hops = num_hops
        self.max_nodes_per_hop = max_nodes_per_hop
        self.node_feat = node_feat
        self.neg_edge_index = neg_edges.t() if neg_edges is not None else None
        self.node_label = node_label
        self.dataset_name = dataset_name

        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'SEAL_data_{}.pt'.format(self.dataset_name)
        return [name]

    def process(self):
        print("Processing dataset.")
        pos_edge, neg_edge = get_pos_neg_edges(self.pos_edge_index, self.num_nodes, self.neg_edge_index)
        edge_weight = torch.ones(self.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.edge_index[0], self.edge_index[1])), 
            shape=(self.num_nodes, self.num_nodes))
        pos_list = extract_enclosing_subgraphs(pos_edge, A, self.node_feat, 1, 
                                                self.num_hops, self.node_label, self.max_nodes_per_hop)
        neg_list = extract_enclosing_subgraphs(neg_edge, A, self.node_feat, 0, 
                                                self.num_hops, self.node_label, self.max_nodes_per_hop)
        
        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list 

@torch.no_grad()
def test(dataset, model, batch_size, args, evaluator):
    model.eval()
    data_loader = DataLoader(dataset, batch_size)
    pbar = tqdm(data_loader, ncols=80)
    y_pred, y_true = [], []
    for data in pbar:
        data = data.to(device)
        x = data.x if args.use_feature else None
        logits = model(data.z, data.edge_index, data.batch, x)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    result = {}
    auc = roc_auc_score(val_true, val_pred)
    print("AUC:{}".format(auc))
    result['AUC'] = auc

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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    print(args)
    
    # Data Preparation
    print("Loading data.")
    dataset_name = args.dataset
    dataset_root = '../../data'
    dataset = PygLinkPropPredDataset(
        name=dataset_name, root=dataset_root)
    evaluator = Evaluator(dataset_name)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    train_pos_edges = split_edge['train']['edge'][:100000]
    val_pos_edge, val_neg_edge = split_edge['valid']['edge'], split_edge['valid']['edge_neg']
    test_pos_edge, test_neg_edge = split_edge['test']['edge'], split_edge['test']['edge_neg']
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seal = SEAL(gnn='SAGE', args=args, max_z=10000)
    train_dataset = SEALDataset(dataset.root+'_seal', graph.edge_index, train_pos_edges, 
                    graph.num_nodes, args.num_hops, args.max_nodes_per_hop, node_feat=graph.x)
    seal.train(train_dataset, device, args)

    # Test
    val_dataset = SEALDataset(dataset.root+"_seal_val", graph.edge_index, val_pos_edge,
                            graph.num_nodes, args.num_hops, args.max_nodes_per_hop,
                            node_feat=graph.x, neg_edges=val_neg_edge)
    test(val_dataset, seal.model, args.batch_size, args, evaluator)