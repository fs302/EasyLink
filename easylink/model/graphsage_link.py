import os
import sys
import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.nn import SAGEConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm

parent_path = os.path.dirname(os.path.dirname(sys.path[0]))
if parent_path not in sys.path:
    sys.path.append(parent_path)
from easylink.model.link_nn import LinkNN
from easylink.common.eval_utils import evaluate_hits, evaluate_auc

class GraphSageLinkPredictor():
    """ Link Prediction based on GraphSage Node Embedding 
    cite: [NIPS2017] [Inductive Representation Learning on Large Graphs](http://snap.stanford.edu/graphsage/)

    """

    def __init__(self, num_nodes, node_features, adj, args):
        self.device = f'cuda:{args.device}' if torch.cuda.is_available(
        ) else 'cpu'
        self.device = torch.device(self.device)

        self.num_nodes = num_nodes
        self.node_features = node_features
        self.adj = adj
        sage_params = {'input_dim': 128,
                       'hidden_channels': 256,
                       'out_dim': 128,
                       'num_layers': 2}
        link_params = {'hidden_channels': args.hidden_channels,
                       'out_channels': 1,
                       'num_layers': args.num_layers,
                       'dropout': args.dropout}
        self.model = SAGE(sage_params['input_dim'],
                          sage_params['hidden_channels'],
                          sage_params['out_dim'],
                          sage_params['num_layers']
                          ).to(self.device)
        self.link_predictor = LinkNN(sage_params['out_dim'],
                                     link_params['hidden_channels'],
                                     link_params['out_channels'],
                                     link_params['num_layers'],
                                     link_params['dropout']
                                     ).to(self.device)

    def train(self, pos_edges, lr=0.001, epochs=10, batch_size=1024, run_validation=False):
        model = self.model
        link_predictor = self.link_predictor
        # initalization
        model.reset_parameters()
        link_predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(link_predictor.parameters()),
            lr=lr
        )
        # training process
        pos_train_edges = pos_edges.to(self.device)
        for epoch in range(1, epochs + 1):
            model.train()
            link_predictor.train()
            data_loader = DataLoader(
                range(pos_train_edges.size(0)), batch_size, shuffle=True)
            pbar = tqdm(data_loader)
            for perm in pbar:
                optimizer.zero_grad()
                edge = pos_train_edges[perm].t()
                h = model(self.node_features, self.adj)
                pos_out = link_predictor(h[edge[0]], h[edge[1]])
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                # can be improved
                edge = torch.randint(
                    0, self.num_nodes, edge.size(), dtype=torch.long, device=self.device)
                neg_out = link_predictor(h[edge[0]], h[edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

                loss = pos_loss + neg_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm(link_predictor.parameters(), 1.0)

                optimizer.step()
                avg_loss = np.round(loss.item(), 6)
                pbar.set_description(
                    "epoch:{epoch}, loss:{avg_loss}".format(**locals()))
            if run_validation:
                self.valication(batch_size)

    def validation(self, batch_size=1024):
        pos_valid_edge = split_edge['valid']['edge'].to(self.device)
        neg_valid_edge = split_edge['valid']['edge_neg'].to(
            self.device)
        pos_test_edge = split_edge['test']['edge'].to(self.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(self.device)

        pos_valid_pred = self.predict(pos_valid_edge, batch_size)
        neg_valid_pred = self.predict(neg_valid_edge, batch_size)
        pos_test_pred = self.predict(pos_test_edge, batch_size)
        neg_test_pred = self.predict(neg_test_edge, batch_size)

        eval_res = evaluate_hits(
            evaluator, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
        for key, result in eval_res.items():
            valid_hits, test_hits = result
            print(key)
            print(
                f'Valid: {100 * valid_hits:.2f}%, '
                f'Test: {100 * test_hits:.2f}%')


    def predict(self, link_list, batch_size=1024):
        preds = []
        self.model.eval()
        self.link_predictor.eval()
        for perm in DataLoader(range(link_list.size(0)), batch_size):
            edge = link_list[perm].t()
            h = self.model(self.node_features, self.adj)
            preds += [self.link_predictor(h[edge[0]],
                                          h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=True):
        super(SAGE, self).init()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSage LinkPredictor')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args
    print(args)

    print("Loading data.")
    dataset_name = 'ogbl-collab'
    dataset = PygLinkPropPredDataset(
        name=dataset_name, root='/home/admin/workspace/project/EasyLink/data')
    data = dataset[0]

    graphsage_linkpredictor = GraphSageLinkPredictor(
        data.num_nodes,
        data.x,
        data.adj_t,
        args
    )

    split_edge = dataset.get_edge_split()
    pos_edges = split_edge['train']['edge']
    evaluator = Evaluator(dataset_name)
    graphsage_linkpredictor.train(pos_edges,
                                  args.lr,
                                  args.epochs,
                                  args.batch_size,
                                  run_validation=True
                                  )
