def test(loader):
#      model.eval()

#      correct = 0
#      for data in loader:  # Iterate in batches over the training/test dataset.
#          out = model(data.x, data.edge_index, data.batch)  
#          pred = out.argmax(dim=1)  # Use the class with highest probability.
#          correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#      return correct / len(loader.dataset)  # Derive ratio of correct predictions.


# for epoch in range(1, 171):
#     train()
#     train_acc = test(train_loader)
#     test_acc = test(test_loader)
#     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
