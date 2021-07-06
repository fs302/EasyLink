import numpy as np
import scipy.sparse as ssp
import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm


def resource_allocation(adj_matrix, link_list, batch_size=32768):
    '''
    cite: [Similarity index based on local paths for link prediction of complex networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.80.046122)
    :param adj_matrix: Compressed Sparse Row matrix
    :param link_list: torch tensor list of links, shape[m, 2]
    :return: RA similarity for each link
    '''
    A = adj_matrix  # e[i, j]
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    D = A.multiply(w).tocsr()  # e[i,j] / log(d_j)

    link_index = link_list.t()
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    scores = []
    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        batch_scores = np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten()
        scores.append(batch_scores)
    scores = np.concatenate(scores, 0)

    return torch.FloatTensor(scores)


def evaluate_hits(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def main():
    # Data settings
    print("Loading data.")
    dataset_name = 'ogbl_ppa'
    dataset = PygLinkPropPredDataset(name=dataset_name)
    evaluator = Evaluator(name=dataset_name)
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # graph_construct
    print("Constructing graph.")
    train_edges_raw = np.array(split_edge['train']['edge'])
    train_edges_reverse = np.array(
        [train_edges_raw[:, 1], train_edges_raw[:, 0]]).transpose()
    train_edges = np.concatenate(
        [train_edges_raw, train_edges_reverse], axis=0)
    edge_weight = torch.ones(train_edges.shape[0], dtype=int)
    A = ssp.csr_matrix(
        (edge_weight, (train_edges[:, 0], train_edges[:, 1])), shape=(
            data.num_nodes, data.num_nodes)
    )

    # test
    print("Benchmark test.")
    batch_size = 1024
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']

    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    model_predictor = resource_allocation  # use model_name as function
    pos_valid_pred = model_predictor(A, pos_valid_edge, batch_size=batch_size)
    neg_valid_pred = model_predictor(A, neg_valid_edge, batch_size=batch_size)

    pos_test_pred = model_predictor(A, pos_test_edge)
    neg_test_pred = model_predictor(A, neg_test_edge)

    eval_res = evaluate_hits(evaluator, pos_valid_pred,
                             neg_valid_pred, pos_test_pred, neg_test_pred)

    for key, result in eval_res.items():
        valid_hits, test_hits = result
        print(key)
        print(
            f'Valid: {100 * valid_hits:.2f}%, '
            f'Test: {100 * test_hits:.2f}%')


if __name__ == '__main__':
    main()
