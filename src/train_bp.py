import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.backprop import MLP

writer = SummaryWriter()


def train_bp(train_loader, val_loader, num_feats, num_pdfs, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLP(num_feats, num_pdfs).to(device)
    print(sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_acc_train = 0.0
    max_acc_val = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            for name, params in model.named_parameters():
                if params.grad is not None and not name.endswith("bias"):
                    writer.add_histogram(f"BP-3Layer_{num_pdfs}/grad_{name}", params.grad, epoch)

            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total
        max_acc_train = max(max_acc_train, train_acc)
        writer.add_scalar(f"BP-3Layer_{num_pdfs}/train_acc", train_acc, epoch)

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
        max_acc_val = max(max_acc_val, val_acc)
        writer.add_scalar(f"BP-3Layer_{num_pdfs}/val_acc", val_acc, epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train CE: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    del model
    torch.cuda.empty_cache()
    return max_acc_train, max_acc_val
