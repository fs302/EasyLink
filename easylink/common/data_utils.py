import sys
import numpy as np
import networkx as nx
import scipy.sparse as ss

def load_basic_network(file_path, graph_type='undirected'):
    g = nx.DiGraph() if graph_type == 'directed' else nx.Graph()
    try:
        with open(file_path) as f:
            print("loading file:", file_path)
            for l in f:
                data = l.strip().split()
                id1 = int(data[0])
                id2 = int(data[1])
                if len(data) > 2:
                    weight = float(data[2])
                else:
                    weight = 1.0
                g.add_edge(id1, id2, weight=weight)
    except IOError:
        print("error:", sys.exc_info()[0])
        raise
    print("#nodes:",g.number_of_nodes(),",#edges:",g.number_of_edges())
    return g

def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def train_test_split(nx_adj, test_frac=.2, val_frac=.1, pos_neg_ratio=.5):

    # Remove diagonal elements
    nx_adj = nx_adj - ss.dia_matrix((nx_adj.diagonal()[np.newaxis, :], [0]), shape=nx_adj.shape)
    nx_adj.eliminate_zeros()

    g = nx.from_scipy_sparse_array(nx_adj)

    adj_triu = ss.triu(nx_adj)
    edges, weights, adj_shape = sparse_to_tuple(adj_triu)

    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    np.random.shuffle(edge_tuples)

    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))
    test_edges = edge_tuples[:num_test]
    val_edges = edge_tuples[num_test:num_test+num_val]
    train_edges = edge_tuples[num_test+num_val:]

    print("Negative Sampling.")
    all_edge_set = set(edge_tuples)
    neg_edge_set = set()
    nodes_set = g.nodes
    while len(neg_edge_set) < len(edge_tuples)/pos_neg_ratio:
        idx_i = np.random.randint(0, nx_adj.shape[0])
        idx_j = np.random.randint(0, nx_adj.shape[0])
        if idx_i == idx_j or idx_i not in nodes_set or idx_j not in nodes_set:
            continue
        neg_edge = (idx_i, idx_j)
        if neg_edge in all_edge_set or neg_edge in neg_edge_set:
            continue
        neg_edge_set.add(neg_edge)

    neg_edge_tuples = list(neg_edge_set)
    num_test_neg = int(num_test/pos_neg_ratio)
    num_val_neg = int(num_val/pos_neg_ratio)
    test_neg_edges = neg_edge_tuples[:num_test_neg]
    val_neg_edges = neg_edge_tuples[num_test_neg:num_test_neg + num_val_neg]
    train_neg_edges = neg_edge_tuples[num_test_neg + num_val_neg:]

    return train_edges, test_edges, val_edges, train_neg_edges, test_neg_edges, val_neg_edges
