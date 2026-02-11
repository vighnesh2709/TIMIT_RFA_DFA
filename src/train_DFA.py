import torch
from tqdm import tqdm
from model.DFA import DFA_MLP
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

prof = profile(
    activities = [ProfilerActivity.CPU],
    record_shapes = True,
    with_flops = True,
)

def train_dfa(X, Y, num_feats, num_pdfs):

    # ------------------ HYPERPARAMS ------------------
    epochs = 2
    batch_size = 256
    lr = 1e-3
    train_ratio = 0.9
    hidden_dim = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X = X.to(device)
    Y = Y.to(device)

    # ------------------ TRAIN / VAL SPLIT ------------------
    N = X.size(0)
    perm = torch.randperm(N, device=device)

    train_N = int(train_ratio * N)
    train_idx = perm[:train_N]
    val_idx   = perm[train_N:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]

    # ------------------ MODEL ------------------
    model = DFA_MLP(num_feats, hidden_dim, num_pdfs).to(device)

    # ------------------ DFA FEEDBACK MATRICES ------------------
    B2 = torch.randn(num_pdfs, hidden_dim, device=device) / hidden_dim**0.5
    B1 = torch.randn(num_pdfs, hidden_dim, device=device) / hidden_dim**0.5

    # ------------------ LOGGING ------------------
    train_ce_hist, val_ce_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    max_acc_train = 0
    max_acc_val = 0
    
    # ================== TRAIN ==================
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
            if epoch == 0 and i == 0:
                prof.start()

            xb = X_shuf[i:i+batch_size]
            yb = Y_shuf[i:i+batch_size]

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

                # -------- DFA BACKWARD (DIRECT) --------
                delta2 = (delta3 @ B2) * (a2 > 0).float()
                delta1 = (delta3 @ B1) * (a1 > 0).float()
                
                #---------VISUALIZING GRADIENTS----------
                writer.add_histogram(f'DFA/delta3_{num_pdfs}', delta3, epoch)
                writer.add_histogram(f'DFA/delta2_{num_pdfs}', delta2, epoch)
                writer.add_histogram(f'DFA/delta1_{num_pdfs}', delta1, epoch)
                
                # -------- WEIGHT UPDATES --------
                model.fc3.weight -= lr * (delta3.T @ h2)
                model.fc2.weight -= lr * (delta2.T @ h1)
                model.fc1.weight -= lr * (delta1.T @ xb)
                

                writer.add_histogram(f'DFA/grad_fc3_{num_pdfs}', delta3.T @ h2, epoch)
                writer.add_histogram(f'DFA/grad_fc2_{num_pdfs}', delta2.T @ h1, epoch)
                writer.add_histogram(f'DFA/grad_fc1_{num_pdfs}', delta1.T @ xb, epoch)


                model.fc3.bias -= lr * delta3.sum(dim=0)
                model.fc2.bias -= lr * delta2.sum(dim=0)
                model.fc1.bias -= lr * delta1.sum(dim=0)

                # -------- ACCURACY --------
                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
            
            if i == 0 and epoch == 0:
                prof.stop()
                total_flops = sum([item.flops for item in prof.key_averages() if item.flops > 0])
                
                write_RFA = open(f"../results/DFA_flops_{num_pdfs}.txt", "w")
                write_RFA.write(f"TOTAL FLOPS: {total_flops}\n")
                write_RFA.write(prof.key_averages().table(sort_by="flops"))
                write_RFA.close() 

        train_acc = correct / total
        train_ce = epoch_loss / (X_train.size(0) / batch_size)
        max_acc_train = max(max_acc_train,train_acc)
        train_acc_hist.append(train_acc)
        train_ce_hist.append(train_ce)
        writer.add_scalar(f"Train/acc_{num_pdfs}",train_acc,epoch)
        # ================== VALIDATION ==================
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
        max_acc_val = max(max_acc_val,val_acc)
        val_acc_hist.append(val_acc)
        val_ce_hist.append(val_ce)
        writer.add_scalar(f"val/acc_{num_pdfs}",val_acc,epoch)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
    torch.cuda.empty_cache()
    del model
    return max_acc_train,max_acc_val


