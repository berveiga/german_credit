"""
Plot train-set and test-set AUC-PR against epochs for an improved NN setup.

This experiment uses Adam, BCEWithLogitsLoss with class weighting,
ReduceLROnPlateau on validation AUC-PR, and validation-based early stopping.
"""

from copy import deepcopy
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "german_credit"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils import Params

PARAMS_PATH = PROJECT_ROOT / "configs/german_credit/params_neuralnet.json"
DATA_PATH = PROJECT_ROOT / "data/raw/german_credit.csv"
PLOT_DIR = PROJECT_ROOT / "plots/german_credit/auc_pr"
OUTPUT_DIR = PROJECT_ROOT / "data/output"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_EPOCHS = 100
EPOCH_STEP = 100
VALIDATION_SIZE = 0.2
ADAM_LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 1e-4
DROPOUT_RATE = 0.1


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

    y_all = input_df_raw["GoodCustomer"]
    df_train_full, df_test = train_test_split(
        input_df_raw,
        test_size=0.33,
        random_state=42,
        stratify=y_all,
    )
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VALIDATION_SIZE,
        random_state=42,
        stratify=df_train_full["GoodCustomer"],
    )

    y_train = df_train["GoodCustomer"].map({-1: 0, 1: 1}).astype("float32")
    x_train = df_train.drop(["GoodCustomer"], axis=1)
    y_val = df_val["GoodCustomer"].map({-1: 0, 1: 1}).astype("float32")
    x_val = df_val.drop(["GoodCustomer"], axis=1)
    y_test = df_test["GoodCustomer"].map({-1: 0, 1: 1}).astype("float32")
    x_test = df_test.drop(["GoodCustomer"], axis=1)

    x_train = pd.get_dummies(x_train, dtype="float32")
    x_val = pd.get_dummies(x_val, dtype="float32")
    x_test = pd.get_dummies(x_test, dtype="float32")
    x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_val = scaler.transform(x_val).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train.to_numpy()).reshape([-1, 1])
    x_val = torch.tensor(x_val)
    y_val = torch.tensor(y_val.to_numpy()).reshape([-1, 1])
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test.to_numpy()).reshape([-1, 1])

    return x_train, y_train, x_val, y_val, x_test, y_test


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

        self.hid1 = nn.Linear(n_input_nodes, 16, bias=True)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.hid2 = nn.Linear(16, 8, bias=True)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.out = nn.Linear(8, 1)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        z = self.dropout1(z)
        z = torch.tanh(self.hid2(z))
        z = self.dropout2(z)
        return self.out(z)


def auc_pr(model, x_data, y_data):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(x_data.to(device))
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().ravel()

    targets = y_data.detach().cpu().numpy().ravel()
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    if was_training:
        model.train()
    return auc(recall, precision)


def positive_class_weight(y_train):
    positives = y_train.sum()
    negatives = len(y_train) - positives
    return (negatives / positives).to(device)


def current_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_german_credit_data()
    train_ds = CreditDataset(x_train, y_train)
    train_ldr = torch.utils.data.DataLoader(
        train_ds, batch_size=params.batch_size, shuffle=True
    )

    model = Net(n_input_nodes=x_train.shape[1]).to(device)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(y_train))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ADAM_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    epoch_values = []
    train_auc_pr_values = []
    val_auc_pr_values = []
    test_auc_pr_values = []
    learning_rates = []

    best_val_auc_pr = -np.inf
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    checks_without_improvement = 0

    print("Using device:", device)
    print(f"Training for up to {params.n_epochs} epochs; recording every {EPOCH_STEP}.")
    print(
        f"Optimizer: Adam(lr={ADAM_LEARNING_RATE}, weight_decay={WEIGHT_DECAY}); "
        f"dropout={DROPOUT_RATE}"
    )

    for epoch in range(1, params.n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_ldr:
            logits = model(x_batch)
            loss = loss_func(logits, y_batch)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch >= MIN_EPOCHS and epoch % EPOCH_STEP == 0:
            train_auc_pr = auc_pr(model, x_train, y_train)
            val_auc_pr = auc_pr(model, x_val, y_val)
            test_auc_pr = auc_pr(model, x_test, y_test)

            scheduler.step(val_auc_pr)
            learning_rate = current_learning_rate(optimizer)

            epoch_values.append(epoch)
            train_auc_pr_values.append(train_auc_pr)
            val_auc_pr_values.append(val_auc_pr)
            test_auc_pr_values.append(test_auc_pr)
            learning_rates.append(learning_rate)

            if val_auc_pr > best_val_auc_pr + MIN_DELTA:
                best_val_auc_pr = val_auc_pr
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            print(
                f"epoch = {epoch:5d} train_loss = {epoch_loss:8.4f} "
                f"train_auc_pr = {train_auc_pr:.4f} "
                f"val_auc_pr = {val_auc_pr:.4f} "
                f"test_auc_pr = {test_auc_pr:.4f} "
                f"lr = {learning_rate:.8f}"
            )

            if checks_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    "Early stopping after "
                    f"{checks_without_improvement} checks without validation gain."
                )
                break

    model.load_state_dict(best_state)
    best_train_auc_pr = auc_pr(model, x_train, y_train)
    best_test_auc_pr = auc_pr(model, x_test, y_test)

    results = pd.DataFrame(
        {
            "epochs": epoch_values,
            "train_auc_pr": train_auc_pr_values,
            "validation_auc_pr": val_auc_pr_values,
            "test_auc_pr": test_auc_pr_values,
            "learning_rate": learning_rates,
        }
    )
    results_path = OUTPUT_DIR / "auc_pr_vs_epochs_optimized.csv"
    plot_path = PLOT_DIR / "auc_pr_vs_epochs_optimized.png"

    results.to_csv(results_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        epoch_values,
        train_auc_pr_values,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Training AUC-PR",
    )
    plt.plot(
        epoch_values,
        test_auc_pr_values,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Test AUC-PR",
    )
    plt.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Best validation epoch ({best_epoch})",
    )
    plt.xlabel("Number of epochs")
    plt.ylabel("AUC-PR")
    plt.title("Training and test AUC-PR vs. epochs, optimized training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    print(f"Best validation AUC-PR: {best_val_auc_pr:.4f} at epoch {best_epoch}")
    print(f"Best-epoch train AUC-PR: {best_train_auc_pr:.4f}")
    print(f"Best-epoch test AUC-PR: {best_test_auc_pr:.4f}")
    print(f"Saved results to {results_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
