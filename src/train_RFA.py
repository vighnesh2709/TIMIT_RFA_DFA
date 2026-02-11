import torch
from tqdm import tqdm
from model.RFA import RFA_MLP
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train_rfa(X, Y, num_feats, num_pdfs,epochs = 20):

    # -------- HYPERPARAMS --------
    batch_size = 256
    lr = 1e-3
    train_ratio = 0.9
    hidden_dim = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X = X.to(device)
    Y = Y.to(device)

    # -------- SPLIT --------
    N = X.size(0)
    #perm = torch.randperm(N, device=device)
    perm = torch.randperm(N)

    train_N = int(train_ratio * N)
    train_idx = perm[:train_N]
    val_idx = perm[train_N:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]

    # -------- MODEL --------
    model = RFA_MLP(num_feats, hidden_dim, num_pdfs).to(device)

    # RFA feedback matrices
    B3 = torch.randn(num_pdfs, hidden_dim, device=device) / hidden_dim**0.5
    B2 = torch.randn(hidden_dim, hidden_dim, device=device) / hidden_dim**0.5


    # -------- TRAINING LOGGING --------
    train_ce_hist, val_ce_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    max_acc_train = 0
    max_acc_val = 0
    global_batch = 0

    # -------- TRAIN --------
    for epoch in range(epochs):

        perm = torch.randperm(X_train.size(0))
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        correct = 0
        total = 0
        epoch_loss = 0.0

        for i in tqdm(
            range(0, X_train.size(0), batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        ):
            xb = X_shuf[i:i+batch_size].to(device)
            yb = Y_shuf[i:i+batch_size].to(device)
           # print(f"xb device: {xb.device}, yb device: {yb.device}, model device: {model.fc1.weight.device}, model bias = {model.fc1.bias.device}")

            with torch.no_grad():

                # -------- FORWARD --------
                a1, h1, a2, h2, logits = model(xb)

                logits = logits - logits.max(dim=1, keepdim=True).values
                probs = torch.softmax(logits, dim=1)

                # -------- ONE-HOT --------
                y_onehot = torch.zeros_like(probs)
                y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                # -------- CE (LOGGING ONLY) --------
                ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()
                epoch_loss += ce.item()

                # -------- OUTPUT ERROR --------
                delta3 = (probs - y_onehot) / xb.size(0)

                # -------- RFA BACKWARD --------
                delta2 = (delta3 @ B3) * (a2 > 0).float()
                delta1 = (delta2 @ B2) * (a1 > 0).float()

                # -------- WEIGHT UPDATES --------
                model.fc3.weight -= lr * (delta3.T @ h2)
                model.fc2.weight -= lr * (delta2.T @ h1)
                model.fc1.weight -= lr * (delta1.T @ xb)

                model.fc3.bias -= lr * delta3.sum(dim=0)
                model.fc2.bias -= lr * delta2.sum(dim=0)
                model.fc1.bias -= lr * delta1.sum(dim=0)

                writer.add_histogram(f"RFA-3Layer_{num_pdfs}/error_prop_1",delta1,epoch)
                writer.add_histogram(f"RFA-3Layer_{num_pdfs}/error_prop_2",delta2,epoch)
                writer.add_histogram(f"RFA-3Layer_{num_pdfs}/error_prop_3",delta3,epoch)

                writer.add_histogram(f"RFA-3Layer_{num_pdfs}/gradient_1",delta1.T @ xb, epoch)
                writer.add_histogram(f"RFA-3Layer_{num_pdfs}/gradient_2",delta2.T @ h1, epoch)
                writer.add_histogram(f"RFA-3Layer_{num_pdfs}/gradient_3",delta3.T @ h2, epoch)

                # -------- ACCURACY --------
                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                
            global_batch += 1

        train_acc = correct / total
        train_ce = epoch_loss / (X_train.size(0) / batch_size)
        max_acc_train = max(max_acc_train, train_acc)
        train_acc_hist.append(train_acc)
        train_ce_hist.append(train_ce)
        writer.add_scalar(f"RFA-3Layer_{num_pdfs}/train_acc",train_acc,epoch)
        # -------- VALIDATION --------
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for i in range(0, X_val.size(0), batch_size):
                xb = X_val[i:i+batch_size]
                yb = Y_val[i:i+batch_size]

                _, _, _, _, logits = model(xb)
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
        writer.add_scalar(f"RFA-3Layer_{num_pdfs}/val_acc",val_acc,epoch)
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )


    return max_acc_train, max_acc_val
