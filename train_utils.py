import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

def train_model_cpu(X, y, nodes, layers, m, batch, epochs, lr):
    device = torch.device("cpu")

    # MLP
    model = nn.Sequential(
        nn.Linear(X.shape[1], nodes),
        nn.ReLU(),
    )

    for _ in range(layers - 2):
        model.append(nn.Linear(nodes, nodes))
        model.append(nn.ReLU())

    model.append(nn.Linear(nodes, m))
    model = model.to(device)

    # dataset
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total_loss = 0
        for a,b in loader:
            a = a.to(device)
            b = b.to(device)
            opt.zero_grad()
            logits = model(a)
            loss = loss_fn(logits,b)
            loss.backward()
            opt.step()
            total_loss += loss.item()*a.size(0)
        print("epoch:", ep+1, "loss:", total_loss/len(ds))

    return model
