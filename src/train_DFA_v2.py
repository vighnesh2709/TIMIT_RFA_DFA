import torch
from model.DFA_v3 import DFA_MLP
from tqdm import tqdm

def train_dfa(X,Y,num_feats,num_pdfs):
    epochs = 65
    batch_size = 256
    lr = 1e-3
    train_ratio = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ",device)

    X = X.to(device)
    Y = Y.to(device)

    N = X.size(0)
    perm = torch.randperm(N)

    train_N = int(train_ratio * N)
    train_idx = perm[:train_N]
    val_idx = perm[train_N:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]

    model = DFA_MLP(num_feats,num_pdfs).to(device)

    B4 = torch.randn(num_pdfs,1024,device = device)/ (1024 ** 0.5)
    B3 = torch.randn(num_pdfs,1024,device = device)/ (1024 ** 0.5)
    B2 = torch.randn(num_pdfs,1024,device = device)/ (1024 ** 0.5)

    train_ce_hist, val_ce_hist = [],[]
    train_acc_hist, val_acc_hist = [],[]
    max_acc_train = 0
    max_acc_val = 0


    for epoch in range(epochs):

        perm = torch.randperm(X_train.size(0))
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        correct = 0
        total = 0
        epoch_loss = 0.0

        for i in tqdm(
            range(0,X_train.size(0),batch_size),
            desc = f"Epoch {epoch+1}/{epochs}",
            leave = False
        ):
            xb = X_shuf[i:i+batch_size]
            yb = Y_shuf[i:i+batch_size]

            with torch.no_grad():

                a1, h1, a2, h2, a3, h3, logits = model(xb)

                logits = logits - logits.max(dim = 1, keepdim = True).values
                probs = torch.softmax(logits, dim = 1)

                y_onehot = torch.zeros_like(probs)
                y_onehot.scatter_(1,yb.unsqueeze(1),1)

                ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim = 1).mean()
                epoch_loss += ce.item()

                delta4 = (probs - y_onehot)/xb.size(0)

                delta3 = (delta4 @ B4) * (a3 > 0).float()
                delta2 = (delta4 @ B3) * (a2 > 0).float()
                delta1 = (delta4 @ B2) * (a1 > 0).float()

                model.fc4.weight -= lr * (delta4.T @ h3)
                model.fc3.weight -= lr * (delta3.T @ h2)
                model.fc2.weight -= lr * (delta2.T @ h1)
                model.fc1.weight -= lr * (delta1.T @ xb)

                model.fc4.bias -= lr * delta4.sum(dim=0)
                model.fc3.bias -= lr * delta3.sum(dim=0)
                model.fc2.bias -= lr * delta2.sum(dim=0)
                model.fc1.bias -= lr * delta1.sum(dim=0)

                # -------- accuracy --------
                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        train_acc = correct / total
        train_ce = epoch_loss / (X_train.size(0) / batch_size)
        max_acc_train = max(max_acc_train, train_acc)
        train_acc_hist.append(train_acc)
        train_ce_hist.append(train_ce)

        # ------------------ VALIDATION ------------------
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for i in range(0, X_val.size(0), batch_size):
                xb = X_val[i:i+batch_size]
                yb = Y_val[i:i+batch_size]

                _, _, _, _, _, _, logits = model(xb)
                probs = torch.softmax(logits, dim=1)

                y_onehot = torch.zeros_like(probs)
                y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()
                val_loss += ce.item()

                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        val_ce = val_loss / (X_val.size(0) / batch_size)
        max_acc_val = max(max_acc_val, val_acc)
        val_acc_hist.append(val_acc)
        val_ce_hist.append(val_ce)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return max_acc_train, max_acc_val
