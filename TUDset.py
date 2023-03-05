import torch
from torch.nn import ReLU, Linear
import torch.nn.functional as F
import torch_geometric
import torchvision
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GATConv, global_mean_pool

dataset = TUDataset(root="data/TUDataset", name="MUTAG")
print(dataset.num_classes)
print(len(dataset))
print(dataset.num_features, dataset.num_node_features, dataset.num_node_attributes)

data = dataset[0]
print(data)
# torch.manual_seed(1234)
dataset = dataset.shuffle(return_perm=False)
train_data = dataset[:150]
test_data = dataset[150:]
train_loader = DataLoader(train_data, batch_size =64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle =True)
for step, data in enumerate(train_loader):
    print(f"Step: {step}")
    print("========")
    print(data)

class GCNModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super(GCNModel, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.lin = Linear(hidden_size, dataset.num_classes)
    
    def forward(self, x, edge_index, batch):
        out = self.conv1(x, edge_index)
        out = ReLU(out)
        out = self.conv2(out)
        out = ReLU(out)
        out = self.conv3(out)

        out = global_mean_pool(out, batch=batch)
        out = F.dropout(out, p= 0.5, training=self.training)
        out = self.lin(out)

model = GCNModel(64)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr =0.01)
classifier = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = classifier(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        print(data.x, data.edge_index, data.batch)

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
