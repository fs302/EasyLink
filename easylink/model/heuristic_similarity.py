import numpy as np
import scipy.sparse as ssp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def common_neighbors(adj_matrix, link_list, batch_size=32768):
    '''
    cite: [The Link Prediction Problem for Social Networks](https://dl.acm.org/doi/pdf/10.1145/956863.956972)
    :param adj_matrix: Compressed Sparse Row matrix
    :param link_list: torch tensor list of links, shape[m, 2]
    :return: CN similarity for each link
    '''
    A = adj_matrix # e[i, j]

    link_index = link_list.t()
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    scores = []
    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        batch_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(batch_scores)
    scores = np.concatenate(scores, 0)

    return torch.FloatTensor(scores)

def adamic_adar(adj_matrix, link_list, batch_size=32768):
    '''
    cite: [Friends and neighbors on the Web](http://social.cs.uiuc.edu/class/cs591kgk/friendsadamic.pdf)
    :param adj_matrix: Compressed Sparse Row matrix
    :param link_list: torch tensor list of links, shape[m, 2]
    :return: AA similarity for each link
    '''
    A = adj_matrix # e[i, j]
    w = 1 / np.log(A.sum(axis=0))
    w[np.isinf(w)] = 0
    D = A.multiply(w).tocsr() # e[i,j] / log(d_j)

    link_index = link_list.t()
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    scores = []
    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        batch_scores = np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten()
        scores.append(batch_scores)
    scores = np.concatenate(scores, 0)

    return torch.FloatTensor(scores)

def resource_allocation(adj_matrix, link_list, batch_size=32768):
    '''
    cite: [Similarity index based on local paths for link prediction of complex networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.80.046122)
    :param adj_matrix: Compressed Sparse Row matrix
    :param link_list: torch tensor list of links, shape[m, 2]
    :return: RA similarity for each link
    '''
    A = adj_matrix # e[i, j]
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    D = A.multiply(w).tocsr() # e[i,j] / log(d_j)

    link_index = link_list.t()
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    scores = []
    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        batch_scores = np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten()
        scores.append(batch_scores)
    scores = np.concatenate(scores, 0)

    return torch.FloatTensor(scores)

def local_path_index(adj_matrix, link_list, ep=0.01, batch_size=32768):
    '''
    cite: [Similarity index based on local paths for link prediction of complex networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.80.046122)
    :param adj_matrix: Compressed Sparse Row matrix
    :param link_list: torch tensor list of links, shape[m, 2]
    :param ep: epsilon to adjust 3 order path weight
    :return: LPI similarity for each link
    '''
    A = adj_matrix # e[i, j] 
    link_index = link_list.t()
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
 
    scores = []
    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx] 
        scores_2d = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores_3d = A[src].dot(A).dot(A[:,dst]).diagonal()
        batch_scores = scores_2d + ep * scores_3d
        scores.append(batch_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)


if __name__=='__main__':
    # toy data
    num_nodes = 6
    edge_index_raw = np.array([[0,1], [1,2], [1,5], [2,3], [2,4], [3,4], [4,5]])
    edge_index_reverse = np.array([edge_index_raw[:,1], edge_index_raw[:,0]]).transpose()
    edge_index = np.concatenate([edge_index_raw, edge_index_reverse], axis=0)
    edge_weight = torch.ones(edge_index.shape[0], dtype=int)
    A = ssp.csr_matrix((edge_weight, (edge_index[:,0], edge_index[:,1])), shape = (num_nodes, num_nodes))

    # test
    link_list = torch.Tensor([[2,1], [2,3], [2,4], [1,2]])
    

    cn_res = common_neighbors(A, link_list)
    print('Common Neighbors:', cn_res)
    
    aa_res = adamic_adar(A, link_list)
    print('Adamic Adar:',aa_res)
	
    ra_res = resource_allocation(A, link_list)
    print('Resource Allocation:',ra_res)
 
    lpi_res = local_path_index(A, link_list, ep=0.01)
    print('Local Path Index:',lpi_res)