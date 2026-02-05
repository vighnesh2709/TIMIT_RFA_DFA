import torch
import torch.nn as nn
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, record_function
from model.backprop import MLP

prof = profile(
    activities = [ProfilerActivity.CPU],
    record_shapes = True,
    with_flops = True,
)

def train_bp(train_loader, val_loader,num_feats,num_pdfs):


    # ---------- MODEL ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLP(num_feats, num_pdfs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # ---------- TRAIN ----------
    epochs = 1
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    max_acc_train = 0
    max_acc_val = 0

    for epoch in range(epochs):
                
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx,(x, y) in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave = False)):
            if batch_idx == 0 and epoch == 0:
                prof.start()

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if epoch == 0 and batch_idx == 0:
                prof.stop()
                write_temp = open(f"bp_flops{num_pdfs}.txt", "w")
                write_temp.write(prof.key_averages().table(sort_by = "flops"))
                # print(prof.key_averages().table(sort_by = "flops"))


        train_loss /= len(train_loader)
        train_acc = correct / total
        max_acc_train = max(max_acc_train,train_acc)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        
        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        max_acc_val = max(max_acc_val,val_acc)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return max_acc_train,max_acc_val

