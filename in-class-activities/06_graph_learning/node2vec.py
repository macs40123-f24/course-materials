# Adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150,
    )
    return acc


@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(z)
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.savefig('node2vec.png')


if __name__ == '__main__':
    # Cora Citation Network Data: https://paperswithcode.com/dataset/cora
    # nodes = publications, edges = citations
    dataset = Planetoid('/project/macs40123/', name='Cora')
    data = dataset[0]

    # Train on GPU if available; takes <=1 min on GPU, ~15 min on CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize Node2Vec model with hyperparameters
    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    ).to(device)
    
    # define data loader (generates batches of rand walks) and optimizer
    # for training Node2Vec model:
    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    # Train for 100 epochs and print progress
    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    # Plot embeddings in 2D, with different colors for 7 classes of articles/nodes
    # Did node2vec learn to distinguish the node classes based on network structure?
    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
    ]
    plot_points(colors)
