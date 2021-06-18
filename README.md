# EasyLink

EasyLink is an Easy-to-use collection of SOTA Link Prediction models on Networks, especially Social Network. It is designed to provide tutorials for beginers, modular implementation on models and comprehensive guidance for choosing the right model.

## Models
| Model               | Paper                                                        | Module |
| ------------------- | ------------------------------------------------------------ | ------ |
| Adamic Adar         | [2003] [Friends and neighbors on the Web](http://social.cs.uiuc.edu/class/cs591kgk/friendsadamic.pdf) |        |
| Local Path Index    | [2009] [Similarity index based on local paths for link prediction of complex networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.80.046122) |        |
| Resource Allocation | [2009] [Predicting missing links via local information](https://arxiv.org/pdf/0901.0553.pdf) |        |
| Node2Vec            | [KDD2016] [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) |        |
| GraphSage           | [NIPS2017] [Inductive Representation Learning on Large Graphs](http://snap.stanford.edu/graphsage/) |        |
| SEAL                | [NIPS2018] [Link prediction based on graph neural networks](https://arxiv.org/abs/1802.09691) |        |


## DataSets

| DataSet     | Description                                                  | Statistics                                           | Node Fea | Edge Fea |
| ----------- | ------------------------------------------------------------ | ---------------------------------------------------- | -------- | -------- |
| USAir       | a network of US Air lines                                    | 332 nodes and 2,126 edges                            |-|-|
| Facebook    | sampled friendship network of Facebook.                      | 4,039 nodes and 88,234 edges                         |-|-|
| Arxiv       | collaboration network generated from arXiv                   | 18,722 nodes and 198,110 edges                       |-|-|
| twitter     | social circles from Twitter                                  | 81,306 nodes and 1,342,310 edges                     |-|-|
| MAG-collab  | subset of the collaboration network between authors from MAG | 235,868	nodes and 1,285,465 edges	            | Yes |-|