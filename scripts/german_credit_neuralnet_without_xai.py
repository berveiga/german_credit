"""
This code fits a neural network on the German credit dataset.
"""

from datetime import datetime
from pathlib import Path
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay

from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "german_credit"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils import Params

params = Params(PROJECT_ROOT / "configs/german_credit/params_neuralnet.json")

np.random.seed(params.seed)
torch.manual_seed(params.seed)

n_epochs = params.n_epochs
device = torch.device("cpu")

PR_CURVE_DIR = PROJECT_ROOT / "plots/german_credit/pr_curve"
OUTPUT_DIR = PROJECT_ROOT / "data/output"
PR_CURVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_df_raw = pd.read_csv(PROJECT_ROOT / "data/raw/german_credit.csv")

all_features = True

if all_features == False:
    plt.title("ROC curve: " + params.model_type)
    input_df_raw = input_df_raw.drop("CriticalAccountOrLoansElsewhere", axis=1)
    params.n_input_nodes = 26
    params.n_nodes_layer_1 = 26
else:
    input_df_raw.rename(
        columns={"CriticalAccountOrLoansElsewhere": "NoHistoryOfDelayedPayments"},
        inplace=True,
    )

df_train, df_test = train_test_split(input_df_raw, test_size=0.33, random_state=42)

ytrain = df_train["GoodCustomer"]
xtrain = df_train.drop(["GoodCustomer"], axis=1)
ytest = df_test["GoodCustomer"]
xtest = df_test.drop(["GoodCustomer"], axis=1)

xtrain = pd.get_dummies(xtrain, dtype="float32")
xtest = pd.get_dummies(xtest, dtype="float32")
xtest = xtest.reindex(columns=xtrain.columns, fill_value=0)
feature_names = xtrain.columns
params.n_nodes_layer1 = xtrain.shape[1]

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

ytrain = ytrain.map({-1: 0, 1: 1})
ytest = ytest.map({-1: 0, 1: 1})

xtrain = xtrain.astype("float32")
ytrain = ytrain.astype("float32")
xtest = xtest.astype("float32")
ytest = ytest.astype("float32")

xtrain = torch.tensor(xtrain, requires_grad=True)
ytrain = torch.tensor(ytrain.to_numpy(), requires_grad=True).reshape([-1, 1])
xtest = torch.tensor(xtest, requires_grad=True)
ytest = torch.tensor(ytest.to_numpy(), requires_grad=True).reshape([-1, 1])


class CreditDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data.to(device)
        self.y_data = y_data.to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx, :]
        y = self.y_data[idx, :]
        return x, y


n_nodes_layer1 = params.n_nodes_layer1
n_nodes_layer2 = params.n_nodes_layer2
n_nodes_layer3 = params.n_nodes_layer3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hid1 = nn.Linear(n_nodes_layer1, n_nodes_layer2, bias=True)
        self.hid2 = nn.Linear(n_nodes_layer2, n_nodes_layer3, bias=True)
        self.out = nn.Linear(n_nodes_layer3, 1)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        z = torch.tanh(self.hid2(z))
        z = torch.sigmoid(self.out(z))
        return z


def metrics(model, ds, thresh=0.5):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(ds)):
        inpts = ds[i][0]
        target = ds[i][1]

        with torch.no_grad():
            p = model(inpts)

        if target > 0.5 and p >= thresh:
            tp += 1
        elif target > 0.5 and p < thresh:
            fp += 1
        elif target < 0.5 and p < thresh:
            tn += 1
        elif target < 0.5 and p >= thresh:
            fn += 1

    n = tp + fp + tn + fn

    if n != len(ds):
        print("FATAL LOGIC ERROR in metrics()")

    accuracy = (tp + tn) / (n * 1.0)
    precision = (1.0 * tp) / (tp + fp)
    recall = (1.0 * tp) / (tp + fn)
    f1 = 2.0 / ((1.0 / precision) + (1.0 / recall))

    return accuracy, precision, recall, f1


print("\nAssessment of credit risk using PyTorch")
torch.manual_seed(1)
np.random.seed(1)

print("\nCreating train and test Datasets")

train_ds = CreditDataset(xtrain, ytrain)
test_ds = CreditDataset(xtest, ytest)

bat_size = params.batch_size

train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)

print("\nCreating binary NN classifier\n")

net = Net().to(device)
net.train()

lrn_rate = params.learning_rate
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lrn_rate)
n_epochs = params.n_epochs
ep_log_interval = 100

print("Loss function: " + str(loss_func))
print("Optimizer: " + str(optimizer.__class__.__name__))
print("Learn rate: " + "%.3f" % lrn_rate)
print("Batch size: " + str(bat_size))
print("Max epochs: " + str(n_epochs))

print("\nStarting training")

start_time = time.time()

for epoch in range(0, n_epochs):
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(train_ldr):
        X = batch[0]
        Y = batch[1]

        oupt = net(X)

        loss_val = loss_func(oupt, Y)
        epoch_loss += loss_val.item()

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    if epoch % ep_log_interval == 0:
        print("epoch = %4d loss = %8.4f" % (epoch, epoch_loss))

elapsed_time = time.time() - start_time
print("Training (time in seconds):", elapsed_time)


net.eval()

"""
7. Calculate the model performance on the test set
"""

with torch.no_grad():
    p_test = net(test_ds.x_data)

target = test_ds.y_data
target_test_np = target.detach().numpy().ravel()
p_test_np = p_test.detach().numpy().ravel()
ytest_np = ytest.detach().numpy().ravel()

precision_nn_test, recall_nn_test, thresholds_nn_test = precision_recall_curve(
    target_test_np, p_test_np
)

auc_precision_recall_test = round(auc(recall_nn_test, precision_nn_test), 3)

print("---------------Assessment of model performance on the test set---------------")
print("AUC-PR (test set):", str(auc_precision_recall_test))

display = PrecisionRecallDisplay.from_predictions(target_test_np, p_test_np, name="")

_ = display.ax_.set_title("Precision-Recall curve: " + params.model_type)

plt.savefig(PR_CURVE_DIR / "pr_curve_neuralnet.png")
plt.show()

f1_scores_test = (
    2 * recall_nn_test * precision_nn_test / (recall_nn_test + precision_nn_test)
)

f1_scores_nn_test = round(np.max(f1_scores_test), 3)

predicted_clf_nn_best_threshold = (p_test_np >= 0.5).astype("int")

print(confusion_matrix(ytest_np, predicted_clf_nn_best_threshold))
sum(confusion_matrix(ytest_np, predicted_clf_nn_best_threshold))

f1_score_nn_test = round(f1_score(ytest_np, predicted_clf_nn_best_threshold), 3)

print(
    "accuracy:",
    round(accuracy_score(ytest_np, predicted_clf_nn_best_threshold), 3),
)

print("best f1-score in the test set: ", f1_score_nn_test)

fpr, tpr, _ = roc_curve(ytest_np, p_test_np)
auc_roc = roc_auc_score(ytest_np, p_test_np)

plt.plot(fpr, tpr, label="data 1, auc=" + str(round(auc_roc, 2)))
plt.title("ROC curve: " + params.model_type)
plt.legend(loc=4)
plt.show()


"""
7. Calculate the model performance on the training set
"""

with torch.no_grad():
    p_train = net(train_ds.x_data)

target = train_ds.y_data
target_train_np = target.detach().numpy().ravel()
p_train_np = p_train.detach().numpy().ravel()

precision_nn_train, recall_nn_train, thresholds_nn_train = precision_recall_curve(
    target_train_np, p_train_np
)

auc_precision_recall_nn_train = auc(recall_nn_train, precision_nn_train)

print("---------------Assessment of model performance on the train set---------------")
print("AUC-PR (train set):", str(auc_precision_recall_nn_train))

f1_scores_nn_train = (
    2 * recall_nn_train * precision_nn_train / (recall_nn_train + precision_nn_train)
)

f1_score_train = round(np.max(f1_scores_nn_train), 3)

print("Best f1-Score in the training set:", f1_score_train)

df_performance = pd.DataFrame(
    {
        "model": [params.model_type],
        "all_features": [all_features],
        "f1_score_train": [f1_score_train],
        "f1_score_test": [f1_score_nn_test],
        "auc_pr_test": [auc_precision_recall_test],
        "time_stamp": [datetime.now().strftime("%d-%m-%Y-%H-%M")],
    }
)

if all_features == True:
    df_performance.to_csv(
        OUTPUT_DIR / "use_case_performance_all.csv",
        mode="a",
        header=False,
        index=False,
    )
    plt.savefig(PR_CURVE_DIR / "pr_curve_neuralnet_features.png")
else:
    df_performance.to_csv(
        OUTPUT_DIR / "use_case_performance_filtered.csv",
        mode="a",
        header=False,
        index=False,
    )
    plt.savefig(PR_CURVE_DIR / "pr_curve_neuralnet_filtered.png")


"""
    8. Generate explanations
    """
