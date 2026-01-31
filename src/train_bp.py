import torch
import torch.nn as nn
from tqdm import tqdm

from model.backprop import MLP

def train_bp(train_loader, val_loader,num_feats,num_pdfs):


    # ---------- MODEL ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLP(num_feats, num_pdfs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # ---------- TRAIN ----------
    epochs = 100
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}"):
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

        train_loss /= len(train_loader)
        train_acc = correct / total

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

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

