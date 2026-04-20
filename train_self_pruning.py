import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Settings (CPU-friendly)
# =========================
BATCH_SIZE = 64
EPOCHS = 5
LAMBDAS = [0.01, 0.05, 0.1]
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 2000
THRESHOLD = 0.1
SOFT_THRESHOLD = 0.3

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Prunable Layer
# =========================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -1.0))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# =========================
# Model
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 128)
        self.fc2 = PrunableLinear(128, 64)
        self.fc3 = PrunableLinear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =========================
# Data
# =========================
transform = transforms.ToTensor()
data_root = "./data"

try:
    train_data = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=transform
    )
    print("CIFAR-10 found locally.")
except Exception:
    print("CIFAR-10 not found locally. Trying to download...")
    train_data = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

train_subset = torch.utils.data.Subset(train_data, range(TRAIN_SAMPLES))
test_subset = torch.utils.data.Subset(test_data, range(TEST_SAMPLES))

train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================
# Helper: collect all gates
# =========================
def get_all_gates(model):
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)
    return np.array(all_gates)

# =========================
# Training
# =========================
def train_model(lam):
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            loss_cls = F.cross_entropy(out, y)

            sparsity_loss = 0.0
            for m in model.modules():
                if isinstance(m, PrunableLinear):
                    gates = torch.sigmoid(m.gate_scores)
                    sparsity_loss += gates.sum()

            loss = loss_cls + lam * sparsity_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Lambda {lam} | Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)}"
                )

        print(f"Lambda {lam} | Epoch {epoch + 1} | Loss {total_loss:.2f}")

    return model

# =========================
# Evaluation
# =========================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total

# =========================
# Gate statistics
# =========================
def gate_stats(model):
    gates = get_all_gates(model)

    sparsity_hard = 100.0 * np.mean(gates < THRESHOLD)
    sparsity_soft = 100.0 * np.mean(gates < SOFT_THRESHOLD)
    mean_gate = float(np.mean(gates))
    min_gate = float(np.min(gates))
    max_gate = float(np.max(gates))

    return sparsity_hard, sparsity_soft, mean_gate, min_gate, max_gate

# =========================
# Save histogram
# =========================
def save_gate_plot(model, lam):
    gates = get_all_gates(model)

    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50)
    plt.title(f"Gate Distribution for Lambda = {lam}")
    plt.xlabel("Gate value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"results/gates_{lam}.png")
    plt.close()

# =========================
# Run experiments
# =========================
os.makedirs("results", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)

results = []

for lam in LAMBDAS:
    print("\n" + "=" * 60)
    print(f"Starting training for lambda = {lam}")
    print("=" * 60)

    model = train_model(lam)

    acc = evaluate(model)
    sparsity_hard, sparsity_soft, mean_gate, min_gate, max_gate = gate_stats(model)

    print(f"Lambda {lam} | Final Accuracy: {acc:.2f}%")
    print(f"Lambda {lam} | Hard Sparsity (< {THRESHOLD}): {sparsity_hard:.2f}%")
    print(f"Lambda {lam} | Soft Sparsity (< {SOFT_THRESHOLD}): {sparsity_soft:.2f}%")
    print(f"Lambda {lam} | Mean Gate Value: {mean_gate:.4f}")
    print(f"Lambda {lam} | Min Gate Value: {min_gate:.4f}")
    print(f"Lambda {lam} | Max Gate Value: {max_gate:.4f}")

    save_gate_plot(model, lam)
    torch.save(model.state_dict(), f"results/checkpoints/model_{lam}.pt")

    results.append([lam, acc, sparsity_hard, sparsity_soft, mean_gate, min_gate, max_gate])

with open("results/results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "lambda",
        "accuracy",
        "hard_sparsity_below_0.1",
        "soft_sparsity_below_0.3",
        "mean_gate",
        "min_gate",
        "max_gate"
    ])
    writer.writerows(results)

print("\nDone. Check the results folder.")