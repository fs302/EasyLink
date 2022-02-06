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
    def __init__(self, gnn, use_feature, learning_rate, hidden_channels, num_layers, max_z, dropout):
        self.model = None
        self.use_feature =use_feature
        if gnn == 'SAGE':
            self.model = SAGE(hidden_channels, num_layers, max_z, dropout)
        parameters = list(self.model.parameters())
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')
        self.optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)

    def train(self, train_dataset, epochs, batch_size, device):
        self.model.train()
        total_loss, cnt = 0., 0
        for epoch in range(epochs):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            pbar = tqdm(train_loader, ncols=80)
            for data in pbar:
                data = data.to(device)
                self.optimizer.zero_grad()
                x = data.x if self.use_feature else None
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
    def __init__(self, root, base_edges, pos_edges, num_nodes, num_hops, max_nodes_per_hop=10, 
                 node_feat=None, neg_edges=None, node_label='drnl', dataset_name='default'):
        self.edge_index = base_edges.t()
        self.pos_edge_index = pos_edges.t()
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
