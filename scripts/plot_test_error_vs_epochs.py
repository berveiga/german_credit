"""
Plot test-set classification error against the number of training epochs.

The model and preprocessing mirror german_credit_neuralnet_without_xai.py.
Training runs once up to 3000 epochs and records test error every 100 epochs.
"""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "german_credit"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils import Params


PARAMS_PATH = PROJECT_ROOT / "configs/german_credit/params_neuralnet.json"
DATA_PATH = PROJECT_ROOT / "data/raw/german_credit.csv"
PLOT_DIR = PROJECT_ROOT / "plots/german_credit/test_error"
OUTPUT_DIR = PROJECT_ROOT / "data/output"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_EPOCHS = 100
MAX_EPOCHS = 3000
EPOCH_STEP = 100
THRESHOLD = 0.5


params = Params(PARAMS_PATH)
np.random.seed(params.seed)
torch.manual_seed(params.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(params.seed)


def load_german_credit_data():
    input_df_raw = pd.read_csv(DATA_PATH)
    input_df_raw.rename(
        columns={"CriticalAccountOrLoansElsewhere": "NoHistoryOfDelayedPayments"},
        inplace=True,
    )

    df_train, df_test = train_test_split(
        input_df_raw, test_size=0.33, random_state=42
    )

    y_train = df_train["GoodCustomer"].map({-1: 0, 1: 1}).astype("float32")
    x_train = df_train.drop(["GoodCustomer"], axis=1)
    y_test = df_test["GoodCustomer"].map({-1: 0, 1: 1}).astype("float32")
    x_test = df_test.drop(["GoodCustomer"], axis=1)

    x_train = pd.get_dummies(x_train, dtype="float32")
    x_test = pd.get_dummies(x_test, dtype="float32")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train.to_numpy()).reshape([-1, 1])
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test.to_numpy()).reshape([-1, 1])

    return x_train, y_train, x_test, y_test


class CreditDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data.to(device)
        self.y_data = y_data.to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx, :], self.y_data[idx, :]


class Net(nn.Module):
    def __init__(self, n_input_nodes):
        super().__init__()

        self.hid1 = nn.Linear(n_input_nodes, params.n_nodes_layer2, bias=True)
        self.hid2 = nn.Linear(params.n_nodes_layer2, params.n_nodes_layer3, bias=True)
        self.out = nn.Linear(params.n_nodes_layer3, 1)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        z = torch.tanh(self.hid2(z))
        return torch.sigmoid(self.out(z))


def test_error(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = (model(x_test.to(device)) >= THRESHOLD).float()
        targets = y_test.to(device)
        error = (predictions != targets).float().mean().item()
    model.train()
    return error


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_german_credit_data()
    train_ds = CreditDataset(x_train, y_train)
    train_ldr = torch.utils.data.DataLoader(
        train_ds, batch_size=params.batch_size, shuffle=True
    )

    model = Net(n_input_nodes=x_train.shape[1]).to(device)
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)

    epoch_values = []
    test_errors = []

    print("Using device:", device)
    print(f"Training for {MAX_EPOCHS} epochs; recording every {EPOCH_STEP} epochs.")

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = 0.0

        for x_batch, y_batch in train_ldr:
            output = model(x_batch)
            loss = loss_func(output, y_batch)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch >= MIN_EPOCHS and epoch % EPOCH_STEP == 0:
            error = test_error(model, x_test, y_test)
            epoch_values.append(epoch)
            test_errors.append(error)
            print(
                f"epoch = {epoch:4d} train_loss = {epoch_loss:8.4f} "
                f"test_error = {error:.4f}"
            )

    results = pd.DataFrame(
        {"epochs": epoch_values, "test_error": test_errors}
    )
    results_path = OUTPUT_DIR / "test_error_vs_epochs.csv"
    plot_path = PLOT_DIR / "test_error_vs_epochs.png"

    results.to_csv(results_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_values, test_errors, marker="o", linewidth=2)
    plt.xlabel("Number of epochs")
    plt.ylabel("Test-set error")
    plt.title("Test-set error vs. number of epochs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    print(f"Saved results to {results_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
