import pytest 
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

def test_loading_graph():
    dataset_name = 'ogbl-collab'
    dataset = PygLinkPropPredDataset(name=dataset_name, root='../data/')
    split_edge = dataset.get_edge_split()
    assert len(split_edge['train']['edge']) > 0
    assert len(split_edge['valid']['edge']) > 0
    assert len(split_edge['test']['edge']) > 0