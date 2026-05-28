# gnn_explorer

Based on https://www.geeksforgeeks.org/deep-learning/what-are-graph-neural-networks/

Step 1: Imports Libraries
We will import pytorch, scikit learn, matplotlib and numpy.

```python
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
```

Step 2 Load the MUTAG dataset
Uses TUDataset which contains many small graphs.
Shuffles and splits into 80% train and 20% test.
NormalizeFeatures scales node features.
loader_train and loader_test yield batches of graphs.

```python
dataset = TUDataset(root='data/TUDataset', name='MUTAG', use_node_attr=False, transform=T.NormalizeFeatures())

dataset = dataset.shuffle()
n = len(dataset)
n_train = int(0.8 * n)
train_dataset = dataset[:n_train]
test_dataset = dataset[n_train:]

print(f"Loaded MUTAG. Total graphs: {len(dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
loader_test = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

Step 3: Define the GNN model
GINConv is a useful graph aggregator for graph classification.
num_layers controls message passing depth.
global_mean_pool pools node embeddings to graph embeddings.
post_mp is an MLP that converts pooled embedding to class logits.
loss() returns NLL loss expecting F.log_softmax outputs.

```python
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.25):
        super(GNNStack, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropou
        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )))
        for _ in range(1, num_layers):
            self.convs.append(pyg_nn.GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)])

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.ones((data.num_nodes, 1), device=edge_index.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lns[i](x)

        emb = x  
        g_emb = pyg_nn.global_mean_pool(emb, batch)
        out = self.post_mp(g_emb)
        return emb, F.log_softmax(out, dim=1)

    def loss(self, pred_logprob, label):
        return F.nll_loss(pred_logprob, label)
```

Step 4: Instantiate model and optimizer
input_dim uses dataset node features.
Move model to device.
Adam optimizer with small weight decay for regularization.

```python
input_dim = max(1, dataset.num_node_features)
num_classes = dataset.num_classes

model = GNNStack(input_dim=input_dim, hidden_dim=64, output_dim=num_classes, num_layers=3, dropout=0.25).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(model)
```

Step 5: Training and evaluation helpers
train_graph_epoch trains for one epoch across batches.
Multiply loss by batch.num_graphs to accumulate correctly.
eval_graph computes accuracy over test batches.
Functions expect batches moved to device.

```python
def train_graph_epoch(loader):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        emb, pred = model(batch) 
        loss = model.loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / total_graphs

@torch.no_grad()
def eval_graph(loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        emb, pred = model(batch)
        pred_label = pred.argmax(dim=1)
        correct += (pred_label == batch.y).sum().item()
        total += batch.num_graphs
    return correct / total
```

Step 6: Run training loop & log metrics
Train for num_epochs, store loss and test accuracy lists.

```python

```

Step 7: Plot training loss and test accuracy
Use these to check convergence and overfitting.

```python
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True); plt.legend()

plt.subplot(1,2,2)
plt.plot(test_scores, label='Test Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Test Accuracy'); plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()
```

Step 8: Get graph embeddings and t-SNE visualization
Run model over all graphs, pool node embeddings to get graph-level embeddings.
Apply t-SNE to reduce to 2D and scatter-plot colored by class.
Clusters indicate separability of learned graph representations.

```python
@torch.no_grad()
def get_graph_embeddings_and_labels():
    model.eval()
    all_embs = []
    all_labels = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        emb, pred = model(batch)   
        g_emb = pyg_nn.global_mean_pool(emb, batch.batch)
        all_embs.append(g_emb.cpu())
        all_labels.append(batch.y.cpu())
    embs = torch.cat(all_embs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return embs, labels

embs, labels = get_graph_embeddings_and_labels()
print("Embeddings shape:", embs.shape, "Labels shape:", labels.shape)

tsne = TSNE(n_components=2, random_state=42, perplexity=20)
emb2 = tsne.fit_transform(embs)

plt.figure(figsize=(7,6))
scatter = plt.scatter(emb2[:,0], emb2[:,1], c=labels, cmap='tab10', s=40)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title('t-SNE of learned graph embeddings')
plt.show()
```
