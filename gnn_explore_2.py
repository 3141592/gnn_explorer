# gnn_explorer

# Based on https://www.geeksforgeeks.org/deep-learning/what-are-graph-neural-networks/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

dataset = TUDataset(root='data/TUDataset', name='MUTAG', use_node_attr=False, transform=T.NormalizeFeatures())

dataset = dataset.shuffle()
n = len(dataset)
n_train = int(0.8 * n)
train_dataset = dataset[:n_train]
test_dataset = dataset[n_train:]

print(f"Loaded MUTAG. Total graphs: {len(dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

for data in test_dataset:
    print("="*40)
    print("="*40)
    print("="*20)
    print(f"data:\n{data}")
    print("="*20)
    print(f"data.x:\n{data.x}")
    print("="*20)
    print(f"data.edge_index:\n{data.edge_index}")
    print("="*20)
    print(f"data.edge_attr:\n:{data.edge_attr}")
    print("="*20)
    print(f"data.y:\n{data.y}")
    print()


