import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------
# MLP Neural Network
# ---------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_out)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------
# Training function
# ---------------------------------------------------
def train_model(model, loader, epochs=10, lr=1e-3):
    device = get_device()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss={total_loss:.4f}")

    print("[OK] Training complete.")
    return model
