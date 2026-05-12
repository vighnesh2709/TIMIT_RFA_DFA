import torch
from tqdm import tqdm
from model.DFA_v2 import DFA_MLP
from torch.utils.tensorboard import SummaryWriter
from utils.helper import get_orthogonal_matrix
from utils.helper import data_pca

#writer = SummaryWriter()
from utils.helper import file_writer
num_pdfs_main = 0
def train_dfa(X, Y, num_feats, num_pdfs, epochs=20):

    num_pdfs_main = num_pdfs
    # -------- HYPERPARAMS --------
    batch_size = 256
    lr = 1e-3
    train_ratio = 0.9
    hidden_dim = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("Using device:", device)

    # -------- TRAIN / VAL SPLIT --------
    N = X.size(0)
    perm = torch.randperm(N)

    train_N = int(train_ratio * N)
    train_idx = perm[:train_N]
    val_idx = perm[train_N:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    # -------- MODEL --------
    model = DFA_MLP(num_feats, num_pdfs).to(device)

    rep = "mono"
    if X.shape[1] == 429:
        rep = "mono"
    else:
        rep = "tri"
    print(f"REP:{rep}")

    # writer = SummaryWriter(f"runs/DFA/4layer/{rep}")


    # -------- DFA FEEDBACK MATRICES --------
    B4 = torch.randn(num_pdfs, hidden_dim, device=device) / (hidden_dim ** 0.5)

    # B3, B2 = get_orthogonal_matrix(num_pdfs, hidden_dim)
    #B3, B2 = data_pca(num_pdfs, hidden_dim)


    # B3 = torch.from_numpy(B3).float().to(device)
    # B2 = torch.from_numpy(B2).float().to(device)

    B3 = torch.randn(num_pdfs, hidden_dim, device=device) / (hidden_dim ** 0.5)
    B2 = torch.randn(num_pdfs, hidden_dim, device=device) / (hidden_dim ** 0.5)


    train_ce_hist, val_ce_hist = [], []
    train_accs, val_accs = [], []
    max_acc_train = 0
    max_acc_val = 0
    global_batch = 0

    # -------- TRAIN --------
    for epoch in range(epochs):

        perm = torch.randperm(X_train.size(0), device=device)
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
            global_batch += 1

            with torch.no_grad():

                # -------- FORWARD --------
                a1, h1, a2, h2, a3, h3, logits = model(xb)

                logits = logits - logits.max(dim=1, keepdim=True).values
                probs = torch.softmax(logits, dim=1)

                # -------- ONE-HOT --------
                y_onehot = torch.zeros_like(probs)
                y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                # -------- CE (LOGGING ONLY) --------
                ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()
                epoch_loss += ce.item()

                # -------- OUTPUT ERROR --------
                delta4 = (probs - y_onehot) / xb.size(0)

                # -------- DFA BACKWARD (DIRECT) --------
                delta3 = (delta4 @ B4) * (a3 > 0).float()
                delta2 = (delta4 @ B3) * (a2 > 0).float()
                delta1 = (delta4 @ B2) * (a1 > 0).float()

                # -------- WEIGHT UPDATES --------
                grad_delta1 = delta1.T @ xb
                grad_delta2 = delta2.T @ h1
                grad_delta3 = delta3.T @ h2
                grad_delta4 = delta4.T @ h3

                model.fc4.weight -= lr * grad_delta4
                model.fc3.weight -= lr * grad_delta3
                model.fc2.weight -= lr * grad_delta2
                model.fc1.weight -= lr * grad_delta1

                model.fc4.bias -= lr * delta4.sum(dim=0)
                model.fc3.bias -= lr * delta3.sum(dim=0)
                model.fc2.bias -= lr * delta2.sum(dim=0)
                model.fc1.bias -= lr * delta1.sum(dim=0)                 
                
                # if global_batch % 50 == 0:

                #     # -------- Error signals (delta) --------
                #     delta_names = ["L1", "L2", "L3", "L4"]
                #     deltas = [delta1, delta2, delta3, delta4]

                #     for name, tensor in zip(delta_names, deltas):
                #         writer.add_histogram(f"Error/Hist/{name}", tensor, global_batch)
                #         writer.add_scalar(f"Error/Mean/{name}", tensor.mean().item(), global_batch)
                #         writer.add_scalar(f"Error/Norm/{name}", torch.norm(tensor).item(), global_batch)

                #     # -------- Weight gradients --------
                #     grad_names = ["L1", "L2", "L3", "L4"]
                #     grad_tensors = [grad_delta1, grad_delta2, grad_delta3, grad_delta4]

                #     for name, tensor in zip(grad_names, grad_tensors):
                #         writer.add_histogram(f"Grad/Hist/{name}", tensor, global_batch)
                #         writer.add_scalar(f"Grad/Mean/{name}", tensor.mean().item(), global_batch)
                #         writer.add_scalar(f"Grad/Norm/{name}", torch.norm(tensor).item(), global_batch)
                

                # -------- ACCURACY --------
                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        train_acc = correct / total
        train_ce = epoch_loss / (X_train.size(0) / batch_size)
        max_acc_train = max(max_acc_train, train_acc)
        train_accs.append(train_acc)
        train_ce_hist.append(train_ce)
        #writer.add_scalar("TrainAcc/Acc", train_acc, epoch)

        # -------- VALIDATION --------
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
        val_accs.append(val_acc)
        val_ce_hist.append(val_ce)
        # writer.add_scalar("ValAcc/Acc", val_acc, epoch)

        # Print the training and validation results
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # Clear memory
    
    file_writer(f"DFA_4Layer_{rep}_train",train_accs)
    file_writer(f"DFA_4Layer_{rep}_val",val_accs)

    del model
    del X_train, X_val, Y_train, Y_val
    torch.cuda.empty_cache()

    return max_acc_train, max_acc_val

