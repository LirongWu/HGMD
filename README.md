#  Hardness-aware GNN-to-MLP Distillation (HGMD)

This is a PyTorch implementation of Hardness-aware GNN-to-MLP Distillation (HGMD), and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy, and ogbn-arxiv)
* Various teacher GNN architectures (GCN, SAGE, GAT) and student MLPs
* GNN sample hardness estimation and hardness-aware subgraph extraction
* Training paradigm for teacher GNNs and student MLPs




## Introduction

To bridge the gaps between powerful Graph Neural Networks (GNNs) and lightweight Multi-Layer Perceptron (MLPs), GNN-to-MLP Knowledge Distillation
(KD) proposes to distill knowledge from a well-trained teacher GNN into a student MLP. In this paper, we revisit the knowledge samples (nodes) in teacher GNNs from the perspective of hardness, and identify that hard sample distillation may be a major performance bottleneck of existing graph KD
algorithms. The GNN-to-MLP KD involves two different types of hardness, one student-free knowledge hardness describing the inherent complexity of GNN knowledge, and the other student-dependent distillation hardness describing the difficulty of teacher-to-student distillation. However, most of the existing work focuses on only one of these aspects or regards them as one thing. This paper proposes a simple yet effective Hardness-aware GNN-to-MLP Distillation (HGMD) framework, which decouples the two hardnesses and estimates them using a non-parametric approach. Finally, two hardness-aware distillation schemes (i.e., HGMD-weight and HGMD-mixup) are further proposed to distill hardness-aware knowledge from teacher GNNs into the corresponding nodes of student MLPs. As non-parametric distillation, HGMD does not involve any additional learnable parameters beyond the student MLPs, but it still outperforms most of the state-of-the-art competitors. HGMD-mixup improves over the vanilla MLPs by 12.95% and outperforms its teacher GNNs by 2.48% averaged over seven real-world datasets.

<p align="center">
  <img src='./figure/framework.PNG' width="1000">
</p>



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


## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@inproceedings{wu2024teach,
  title={Teach Harder, Learn Poorer: Rethinking Hard Sample Distillation for GNN-to-MLP Knowledge Distillation},
  author={Wu, Lirong and Liu, Yunfan and Lin, Haitao and Huang, Yufei and Li, Stan Z},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={2554--2563},
  year={2024}
}
```


## License

Hardness-aware GNN-to-MLP Distillation (HGMD) is released under the MIT license.