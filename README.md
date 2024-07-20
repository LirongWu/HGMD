#  Hardness-aware GNN-to-MLP Distillation (HGMD)

This is a PyTorch implementation of Hardness-aware GNN-to-MLP Distillation (HGMD), and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy, and ogbn-arxiv)

* Various teacher GNN architectures (GCN, SAGE, GAT) and student MLPs

* GNN sample hardness estimation and hardness-aware subgraph extraction

* Training paradigm for teacher GNNs and student MLPs

  

## Main Requirements

* torch==1.6.0
* dgl == 0.6.1
* scipy==1.7.3
* numpy==1.21.5



## Description

* train_and_eval.py  
  * train_teacher() -- Pre-train the teacher GNNs
  * train_student() -- Train the student MLPs with the pre-trained teacher GNNs
* models.py  
  
  * MLP() -- student MLPs
  * GCN() -- GCN Classifier, working as teacher GNNs
  * GAT() -- GAT Classifier, working as teacher GNNs
  * GraphSAGE() -- GraphSAGE Classifier, working as teacher GNNs
  * cal_weighted_coefficient() -- Calculate mixup coefficients
* dataloader.py  

  * load_data() -- Load Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy, and ogbn-arxiv datasets
* utils.py  
  * set_seed() -- Set radom seeds for reproducible results
  * subgraph_extractor() -- Extract hardness-aware subgraphs based on GNN sample hardness




## Running the code

1. Install the required dependency packages

3. To get the results on a specific *dataset* with specific *GNN* as the teacher, please run with proper hyperparameters:

  ```
python main.py --dataset data_name --teacher gnn_name --model_mode model_mode
  ```

where (1) *data_name* is one of the seven datasets: Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy  ogbn-arxiv; (2) *gnn_name* is one of the three GNN architectures: GCN, SAGE, and GAT; (3) *model_mode* is one of the three schemes: 0 (vanilla), 1 (HGMD-mixup) and 2 (HGMD-weight). Take the HGMD-mixup model with GCN as the teacher model on the Citeseer dataset as an example: 

```
python main.py --dataset citeseer --teacher GCN --model_mode 1
```



## License

Hardness-aware GNN-to-MLP Distillation (HGMD) is released under the MIT license.