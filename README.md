# EasyLink

EasyLink is an **Easy-to-use** collection of SOTA Link Prediction models for Networks, especially Social Networks. It is designed to provide tutorials for beginers, pipeline implementation of models and comprehensive guidance for choosing the right model.

## Requirements
* Python 3.6.2
* PyTorch 1.7.1
* PyTorch_Geometric 1.6.3
* OGB 1.3.1
* tqdm 4.36.1

## Models
| Model               | Paper                                                        | Module |
| ------------------- | ------------------------------------------------------------ | ------ |
| Common Neighbors | [2003] [The Link Prediction Problem for Social Networks](https://dl.acm.org/doi/pdf/10.1145/956863.956972) | [Model](https://github.com/fs302/EasyLink/blob/main/easylink/model/heuristic_similarity.py#L7) /[Example](https://github.com/fs302/EasyLink/blob/main/easylink/example/ogbl_heuristic_pipe.py)  |
| Adamic Adar         | [2003] [Friends and neighbors on the Web](http://social.cs.uiuc.edu/class/cs591kgk/friendsadamic.pdf) |   [Model](https://github.com/fs302/EasyLink/blob/main/easylink/model/heuristic_similarity.py#L27)  /[Example](https://github.com/fs302/EasyLink/blob/main/easylink/example/ogbl_heuristic_pipe.py)   |
| Resource Allocation | [2009] [Predicting missing links via local information](https://arxiv.org/pdf/0901.0553.pdf) |    [Model](https://github.com/fs302/EasyLink/blob/main/easylink/model/heuristic_similarity.py#L50)  /[Example](https://github.com/fs302/EasyLink/blob/main/easylink/example/ogbl_heuristic_pipe.py)  |
| Local Path Index    | [2009] [Similarity index based on local paths for link prediction of complex networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.80.046122) |  [Model](https://github.com/fs302/EasyLink/blob/main/easylink/model/heuristic_similarity.py#L73) /[Example](https://github.com/fs302/EasyLink/blob/main/easylink/example/ogbl_heuristic_pipe.py)|
| Node2Vec            | [KDD2016] [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) |  [Model](https://github.com/fs302/EasyLink/blob/main/easylink/model/node2vec_link.py) |
| GraphSage           | [NIPS2017] [Inductive Representation Learning on Large Graphs](http://snap.stanford.edu/graphsage/) |   [Model](https://github.com/fs302/EasyLink/blob/main/easylink/model/graphsage_link.py)     |
| SEAL                | [NIPS2018] [Link prediction based on graph neural networks](https://arxiv.org/abs/1802.09691) |   [Model](https://github.com/fs302/EasyLink/blob/main/easylink/model/seal.py)  /[Example](https://github.com/fs302/EasyLink/blob/main/easylink/example/seal_pipe.py) |


## DataSets

| DataSet     | Description                                                  | Statistics                                           | Node Fea | Edge Fea |
| ----------- | ------------------------------------------------------------ | ---------------------------------------------------- | -------- | -------- |
| USAir       | a network of US Air lines                                    | 332 nodes and 2,126 edges                            |-|-|
| Facebook    | sampled friendship network of Facebook.                      | 4,039 nodes and 88,234 edges                         |-|-|
| Arxiv       | collaboration network generated from arXiv                   | 18,722 nodes and 198,110 edges                       |-|-|
| twitter     | social circles from Twitter                                  | 81,306 nodes and 1,342,310 edges                     |-|-|
| MAG-collab  | subset of the collaboration network between authors from MAG | 235,868	nodes and 1,285,465 edges	            | Yes |-|