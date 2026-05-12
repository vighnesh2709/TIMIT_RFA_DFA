import torch
from model.RFA_v2 import RFA_MLP
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
from utils.helper import get_orthogonal_matrix
from utils.helper import file_writer
from utils.helper import orthogonal_pair
num_pdfs_main = 0
def train_rfa(X, Y, num_feats, num_pdfs, epochs=20):

    num_pdfs_main = num_pdfs
    # -------- HYPERPARAMS --------
    batch_size = 256
    lr = 1e-3
    train_ratio = 0.9
    hidden_dim = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("Using device:", device)

    # -------- SPLIT --------
    N = X.size(0)
    perm = torch.randperm(N)

    train_N = int(train_ratio * N)
    train_idx = perm[:train_N]
    val_idx = perm[train_N:]

    X_train, Y_train = X[train_idx].to(device), Y[train_idx].to(device)
    X_val, Y_val = X[val_idx].to(device), Y[val_idx].to(device)

    # -------- MODEL --------
    model = RFA_MLP(num_feats, num_pdfs).to(device)

    # -------- RFA FEEDBACK MATRICES --------
    B4 = torch.randn(num_pdfs, hidden_dim, device=device) / (hidden_dim ** 0.5)
    #B3, B2 = orthogonal_pair()

    #B3 = torch.from_numpy(B3).float().to(device)
    #B2 = torch.from_numpy(B2).float().to(device)


    #B3 = torch.from_numpy(B3).float().to(device)
    #B2 = torch.from_numpy(B2).float().to(device)

    B3 = torch.randn(hidden_dim, hidden_dim, device=device) / (hidden_dim ** 0.5)
    B2 = torch.randn(hidden_dim, hidden_dim, device=device) / (hidden_dim ** 0.5)

    
    print("SHAPE OF B3: ", B3.shape)
    print("SHAPE OF B2: ", B2.shape)


    rep = "mono"
    if X.shape[1] == 429:
        rep = "mono"
    else:
        rep = "tri"
    print(f"REP:{rep}")

    # writer = SummaryWriter(f"runs/RFA/4layer/{rep}")



    max_acc_train = 0
    max_acc_val = 0
    global_batch = 0
    train_accs, val_accs = [],[]

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

            xb = X_shuf[i:i+batch_size]
            yb = Y_shuf[i:i+batch_size]

            with torch.no_grad():

                # -------- FORWARD --------
                a1, h1, a2, h2, a3, h3, logits = model(xb)

                logits = logits - logits.max(dim=1, keepdim=True).values
                probs = torch.softmax(logits, dim=1)

                # -------- ONE-HOT --------
                y_onehot = torch.zeros_like(probs)
                y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                # -------- CE (logging only) --------
                ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()
                epoch_loss += ce.item()

                # -------- OUTPUT ERROR --------
                delta4 = (probs - y_onehot) / xb.size(0)

                # -------- RFA BACKWARD --------
                delta3 = (delta4 @ B4) * (a3 > 0).float()
                delta2 = (delta3 @ B3) * (a2 > 0).float()
                delta1 = (delta2 @ B2) * (a1 > 0).float()

                # -------- WEIGHT GRADIENTS --------
                grad1 = delta1.T @ xb
                grad2 = delta2.T @ h1
                grad3 = delta3.T @ h2
                grad4 = delta4.T @ h3

                # -------- UPDATE --------
                model.fc4.weight -= lr * grad4
                model.fc3.weight -= lr * grad3
                model.fc2.weight -= lr * grad2
                model.fc1.weight -= lr * grad1

                model.fc4.bias -= lr * delta4.sum(dim=0)
                model.fc3.bias -= lr * delta3.sum(dim=0)
                model.fc2.bias -= lr * delta2.sum(dim=0)
                model.fc1.bias -= lr * delta1.sum(dim=0)

                # -------- LOGGING (every 50 batches) --------
                # if global_batch % 50 == 0:

                #     # Error signals
                #     deltas = [delta1, delta2, delta3, delta4]
                #     for idx, tensor in enumerate(deltas, 1):
                #         writer.add_histogram(f"Error/Hist/L{idx}", tensor, global_batch)
                #         writer.add_scalar(f"Error/Mean/L{idx}", tensor.mean().item(), global_batch)
                #         writer.add_scalar(f"Error/Norm/L{idx}", torch.norm(tensor).item(), global_batch)

                #     # Gradients
                #     grads = [grad1, grad2, grad3, grad4]
                #     for idx, tensor in enumerate(grads, 1):
                #         writer.add_histogram(f"Grad/Hist/L{idx}", tensor, global_batch)
                #         writer.add_scalar(f"Grad/Mean/L{idx}", tensor.mean().item(), global_batch)
                #         writer.add_scalar(f"Grad/Norm/L{idx}", torch.norm(tensor).item(), global_batch)
                
                # -------- ACCURACY --------
                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            global_batch += 1

        # -------- TRAIN METRICS --------
        train_acc = correct / total
        train_accs.append(train_acc)
        num_batches = math.ceil(X_train.size(0) / batch_size)
        train_ce = epoch_loss / num_batches
        
        # writer.add_scalar("TrainAcc/Acc",train_acc, epoch)

        max_acc_train = max(max_acc_train, train_acc)

        

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
        num_val_batches = math.ceil(X_val.size(0) / batch_size)
        val_ce = val_loss / num_val_batches
        val_accs.append(val_acc)
        max_acc_val = max(max_acc_val, val_acc)

        # writer.add_scalar("ValAcc/Acc",val_acc, epoch)


        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
    file_writer(f"RFA_4Layer_{rep}_train",train_accs)
    file_writer(f"RFA_4Layer_{rep}_val",val_accs)

    #writer.close()
    del model
    torch.cuda.empty_cache()

    return max_acc_train, max_acc_val
