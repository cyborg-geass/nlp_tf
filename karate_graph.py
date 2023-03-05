import os
import torch, torchvision
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy, relu, tanh
from torch.nn.functional import linear
from torch.optim import Adam

# print(torch.__version__)
"""
edge_index: [2, num_of_edges]
x: [num_of_nodes, num_of_features per node]
y: [num_of_nodes to classify]
train_musk: [num_of_nodes already classified]
N: no_of_classifiers (here 4)
"""

def visualize_embedding(h, color, loss= None, epoch = None):
    h = h.detach().cpu().numpy()
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(h[:,0], h[:,1], s=140, color= color,cmap="Set2")
    if loss is not None and epoch is not None:
        plt.xlabel(f"loss value {loss:.4f} for epoch{epoch:3d}")
dataset = KarateClub()
data = dataset[0]
# print(len(dataset))
# print(dataset.num_features)
# print(dataset.num_features)

print(data.is_undirected)
print(data.has_isolated_nodes)
print(data.has_self_loops)
print(data.num_classes)
print(data.num_features, data.num_edge_features)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.linear = torch.nn.Linear(2, dataset.num_classes)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = tanh(h)
        h = self.conv2(h)
        h = tanh(h)
        h = self.conv3(h)
        out = self.linear(h)

        return out, h
model = GCN()
classifier = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters, lr =0.01)

def train():
    model.train(True)
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = classifier(out[data.edge_attr], data.y[data.edge_attr])
    loss.backward()
    optimizer.step()
    return loss, h
for epoch in range(201):
    loss, h =train()
    visualize_embedding(h, color=data.y)