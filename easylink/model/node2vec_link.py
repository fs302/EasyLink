import os, sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.nn import Node2Vec
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm

parent_path = os.path.dirname(os.path.dirname(sys.path[0]))
if parent_path not in sys.path:
    sys.path.append(parent_path)
from easylink.common.eval_utils import evaluate_hits, evaluate_auc
from easylink.model.link_nn import LinkNN

class Node2VecLinkPredictor():
    """ Link Prediction via Node2vec Node
    cite: [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

    This module will project node into embedding based on graph structure
     and train an nn model for node-pair embedding product based on training sample.

    Returns:
        [type]: [description]
    """

    def __init__(self, edge_index, emb_path='n2v_emb.pt', n2v_params=None, loading_pretrain=False):
        self.edge_index = edge_index
        self.emb_path = emb_path
        self.device = 0

        default_n2v_params = {'embedding_dim': 128,
                              'walk_length': 40,
                              'context_size': 20,
                              'walks_per_node': 10,
                              'p': 1.0,
                              'q': 1.0,
                              'num_negative_samples': 10,
                              'batch_size': 256,
                              'num_workers': 4,
                              'lr': 1e-3,
                              'epochs': 2}
        if n2v_params is None:
            n2v_params = default_n2v_params
        else:
            # complete with default
            for k, v in default_n2v_params.items():
                if k not in n2v_params:
                    n2v_params[k] = v
        self.embedding_dim = n2v_params['embedding_dim']
        self.walk_length = n2v_params['walk_length']
        self.context_size = n2v_params['context_size']
        self.walks_per_node = n2v_params['walks_per_node']
        self.n2v_batch_size = n2v_params['batch_size']
        self.n2v_num_workers = n2v_params['num_workers']
        self.n2v_lr = n2v_params['lr']
        self.n2v_epochs = n2v_params['epochs']
        self.p = n2v_params['p']
        self.q = n2v_params['q']
        self.num_negative_samples = n2v_params['num_negative_samples']

        self.embedding = None
        if loading_pretrain:
            self.load_embedding(self.emb_path)

        self.predictor = None

    def load_embedding(self, emb_path):
        print("Loading embedding.")
        self.embedding = torch.load(emb_path, map_location='cpu')

    def save_embedding(self, model, emb_path):
        torch.save(model.embedding.weight.data.cpu(), emb_path)

    def train_node2vec(self, store_embedding=False):
        print("Training Node2vec embedding.")
        device = f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model = Node2Vec(self.edge_index, self.embedding_dim, self.walk_length, 
                         self.context_size, self.walks_per_node, self.p, self.q, 
                         self.num_negative_samples, sparse=True).to(device)
        loader = model.loader(batch_size=self.n2v_batch_size,
                              shuffle=True, num_workers=self.n2v_num_workers)
        optimizer = torch.optim.SparseAdam(
            list(model.parameters()), lr=self.n2v_lr)

        model.train()
        log_steps = 10
        for epoch in range(1, self.n2v_epochs + 1):
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

                # TODO:log
                if (i + 1) % log_steps == 0:
                    print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                          f'Loss: {loss:.4f}')
            if store_embedding:
                self.save_embedding(model, self.emb_path)

        self.embedding = model.embedding.weight.data.cpu()
        return model

    def train_link_predictor(self, pos_edges, hidden_channels, num_layers, dropout=True, epochs=10, batch_size=1024, run_validation=False):
        print("Training Link Predictor.")
        device = f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        x = self.embedding.to(device)
        pos_edges = pos_edges.to(device)

        predictor = LinkNN(x.size(-1), hidden_channels, 1,
                           num_layers, dropout).to(device)

        
        predictor.reset_parameters()
        lr = 0.001
        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            predictor.train()
            total_loss = 0
            data_loader = DataLoader(
                range(pos_edges.size(0)), batch_size, shuffle=True)
            pbar = tqdm(data_loader)
            for perm in pbar:
                optimizer.zero_grad()
                edge = pos_edges[perm].t()
                pos_out = predictor(x[edge[0]], x[edge[1]])
                pos_loss = -torch.log(pos_out + 1e-15).mean()

                edge = torch.randint(
                    0, x.size(0), edge.size(), dtype=torch.long, device=x.device)
                neg_out = predictor(x[edge[0]], x[edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

                loss = pos_loss + neg_loss
                loss.backward()
                optimizer.step()
                avg_loss = np.round(loss.item(),6)
                pbar.set_description(
                    "epoch:{epoch}, loss:{avg_loss}".format(**locals()))
            if run_validation:
                predictor.eval()

                pos_valid_edge = split_edge['valid']['edge'].to(x.device)
                neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
                pos_test_edge = split_edge['test']['edge'].to(x.device)
                neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

                pos_valid_pred = self.predict(x, pos_valid_edge, batch_size, predictor)
                neg_valid_pred = self.predict(x, neg_valid_edge, batch_size, predictor)
                pos_test_pred = self.predict(x, pos_test_edge, batch_size, predictor)
                neg_test_pred = self.predict(x, neg_test_edge, batch_size, predictor)

                eval_res = evaluate_hits(evaluator, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
                for key, result in eval_res.items():
                    valid_hits, test_hits = result
                    print(key)
                    print(
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%')
        predictor.eval()
        self.predictor = predictor
        return predictor

    def predict(self, x, link_list, batch_size=1024, predictor=None):
        preds = []
        if predictor is None:
            predictor = self.predictor
        predictor.eval()
        for perm in DataLoader(range(link_list.size(0)), batch_size):
            edge = link_list[perm].t()
            preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Node2Vec LinkPredictor')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--n2v_embedding_dim', type=int, default=64)
    parser.add_argument('--n2v_walk_length', type=int, default=40)
    parser.add_argument('--n2v_context_size', type=int, default=20)
    parser.add_argument('--n2v_walks_per_node', type=int, default=10)
    parser.add_argument('--n2v_num_workers', type=int, default=4)
    parser.add_argument('--n2v_batch_size', type=int, default=1024)
    parser.add_argument('--n2v_train_epochs', type=int, default=10)
    parser.add_argument('--n2v_lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    print(args)
    # test
    print("Loading data.")
    dataset_name = 'ogbl-collab'
    dataset = PygLinkPropPredDataset(
        name=dataset_name, root='/home/admin/workspace/project/EasyLink/data')
    data = dataset[0]

    emb_path = '/home/admin/workspace/project/EasyLink/easylink/model/n2v_emb/test.pt'
    n2v_params = {'embedding_dim': args.n2v_embedding_dim,
                    'walk_length': args.n2v_walk_length,
                    'context_size': args.n2v_context_size,
                    'walks_per_node': args.n2v_walks_per_node,
                    'batch_size': args.n2v_batch_size,
                    'num_workers': args.n2v_num_workers,
                    'lr': args.n2v_lr,
                    'epochs': args.n2v_train_epochs}
    
    n2v = Node2VecLinkPredictor(
        data.edge_index, emb_path, n2v_params, loading_pretrain=True)
    # n2v.train_node2vec(store_embedding=True)

    split_edge = dataset.get_edge_split()
    pos_edges = split_edge['train']['edge']
    evaluator = Evaluator(dataset_name)
    n2v.train_link_predictor(pos_edges, args.hidden_channels,
                             args.num_layers, args.dropout, args.epochs, args.batch_size, run_validation=True)
