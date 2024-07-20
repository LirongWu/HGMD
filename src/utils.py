import os
import dgl
import torch
import shutil
import random
import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy

import matplotlib
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def get_evaluator(dataset):
    def evaluator(out, labels):
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()

    return evaluator


def extract_indices(g):
    edge_idx_loop = g.adjacency_matrix(transpose=True)._indices()
    edge_idx_no_loop = dgl.remove_self_loop(g).adjacency_matrix(transpose=True)._indices()
    edge_idx = (edge_idx_loop, edge_idx_no_loop)
    
    return edge_idx


def subgraph_extractor(out_t, out_s, edge_list, alpha):
    out_t = out_t.softmax(dim=-1).detach().cpu().numpy()
    out_s = out_s.softmax(dim=-1).detach().cpu().numpy()
    
    # Cora, Citeseer, Pubmed: No special Operations
    entropy_t = entropy(out_t, axis=-1)
    entropy_t = entropy_t / np.max(entropy_t)
    entropy_t[entropy_t==0] = 1e-8
    entropy_s = entropy(out_s, axis=-1)
    entropy_s = entropy_s / np.max(entropy_s)
    entropy_s[entropy_s==0] = 1e-8

    # Calculate sample hardness
    similarity_t = np.sum(out_t[edge_list[0]] * out_t[edge_list[1]], axis=-1) / (norm(out_t[edge_list[0]], axis=-1) * norm(out_t[edge_list[1]], axis=-1))
    edge_prob = 1 - np.exp(-alpha * similarity_t * np.sqrt(entropy_t[edge_list[0]] * entropy_s[edge_list[0]]) / entropy_t[edge_list[1]])  

    return edge_prob