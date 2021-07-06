import torch
from torch_geometric.nn import Node2Vec

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

class Node2VecLinkPredictor():
    """ Link Prediction via Node2vec Node
    cite: [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

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

        self.embedding = None
        if loading_pretrain:
            self.load_embedding(self.emb_path)

    def load_embedding(self, emb_path):
        self.embedding = torch.load(emb_path, map_location='cpu')

    def save_embedding(self, model, emb_path):
        torch.save(model.embedding.weight.data.cpu(), emb_path)

    def train_node2vec(self):

        device = f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model = Node2Vec(self.edge_index, self.embedding_dim, self.walk_length,
                         self.context_size, self.walks_per_node, sparse=True).to(device)
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

            self.save_embedding(model, self.emb_path)
        
        self.embedding = model.embedding.weight.data.cpu()
        return model

    def train_link_predictor(self, hidden_channels, num_layers, dropout=True):
        pass

    def predict(self, link_list):
        pass


if __name__ == '__main__':
    # test
    print("Loading data.")
    dataset = PygLinkPropPredDataset(name='ogbl-collab', root='/home/admin/workspace/project/EasyLink/data')
    data = dataset[0]
    
    print("Training Node2vec embedding.")
    emb_path = 'n2v_emb/test.pt'
    n2v_params = {"epochs": 10}
    n2v = Node2VecLinkPredictor(data.edge_index, emb_path, n2v_params)
    n2v.train_node2vec()

