import torch
from model.RFA_v2 import RFA_MLP
from tqdm import tqdm


class RawWeightLogger:
    """Log raw weight values after every update (no truncation)"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w')
        
        # Write header
        self.file.write("="*100 + "\n")
        self.file.write("RAW WEIGHT VALUES LOG (Per Update - Complete Values)\n")
        self.file.write("="*100 + "\n\n")
        self.file.flush()
    
    def log_batch(self, batch_num, model):
        """Log raw weight values for current batch (no truncation)"""
        
        self.file.write(f"\n{'='*100}\n")
        self.file.write(f"BATCH {batch_num}\n")
        self.file.write(f"{'='*100}\n\n")
        
        for name, param in model.named_parameters():
            if param.data is not None:
                weight = param.data
                
                self.file.write(f"{name}:\n")
                self.file.write(f"  Shape: {weight.shape}\n")
                self.file.write(f"  Values:\n")
                
                # Use torch.set_printoptions to print all values
                torch.set_printoptions(profile='full', linewidth=120, sci_mode=False)
                weight_str = str(weight)
                
                # Indent the weight string
                for line in weight_str.split('\n'):
                    self.file.write(f"    {line}\n")
                
                self.file.write("\n")
        
        self.file.flush()
    
    def close(self):
        """Close the log file"""
        self.file.write("="*100 + "\n")
        self.file.close()


class WeightStatisticsLogger:
    """Log weight statistics with update deltas"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w')
        self.previous_weights = {}
        
        # Write header
        self.file.write("="*120 + "\n")
        self.file.write("WEIGHT STATISTICS LOG (With Update Deltas)\n")
        self.file.write("="*120 + "\n\n")
        self.file.write(f"{'Batch':<8} {'Layer':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Abs Max':<12} {'Norm':<12} {'Delta Mean':<12} {'Delta Std':<12} {'Delta Max':<12}\n")
        self.file.write("-"*120 + "\n")
        self.file.flush()
    
    def log_batch(self, batch_num, model):
        """Log weight statistics and deltas from previous batch"""
        
        for name, param in model.named_parameters():
            if param.data is not None:
                weight = param.data
                
                # Calculate statistics
                mean = weight.mean().item()
                std = weight.std().item()
                min_val = weight.min().item()
                max_val = weight.max().item()
                abs_max = weight.abs().max().item()
                norm = weight.norm().item()
                
                # Calculate delta (update amount)
                if name in self.previous_weights:
                    delta = weight - self.previous_weights[name]
                    delta_mean = delta.mean().item()
                    delta_std = delta.std().item()
                    delta_max = delta.abs().max().item()
                else:
                    # First batch has no previous weights
                    delta_mean = 0.0
                    delta_std = 0.0
                    delta_max = 0.0
                
                # Store current weights for next iteration
                self.previous_weights[name] = weight.clone().detach()
                
                # Write to file
                self.file.write(
                    f"{batch_num:<8} {name:<15} {mean:<12.6f} {std:<12.6f} {min_val:<12.6f} {max_val:<12.6f} {abs_max:<12.6f} {norm:<12.6f} {delta_mean:<12.6f} {delta_std:<12.6f} {delta_max:<12.6f}\n"
                )
        
        self.file.flush()
    
    def close(self):
        """Close the log file"""
        self.file.write("="*120 + "\n")
        self.file.close()


def train_rfa(X, Y, num_feats, num_pdfs):

    # -------- HYPERPARAMS --------
    epochs = 20
    batch_size = 256
    lr = 1e-3
    train_ratio = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X = X.to(device)
    Y = Y.to(device)

    # -------- SPLIT --------
    N = X.size(0)
    perm = torch.randperm(N, device=device)

    train_N = int(train_ratio * N)
    train_idx = perm[:train_N]
    val_idx = perm[train_N:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]

    # -------- MODEL --------
    model = RFA_MLP(num_feats, num_pdfs).to(device)

    # -------- RFA FEEDBACK MATRICES (4-layer) --------
    # B4: output layer feedback (num_pdfs -> 1024)
    # B3: hidden layer 3 feedback (1024 -> 1024)
    # B2: hidden layer 2 feedback (1024 -> 1024)
    B4 = torch.randn(num_pdfs, 1024, device=device) / (1024**0.5)
    B3 = torch.randn(1024, 1024, device=device) / (1024**0.5)
    B2 = torch.randn(1024, 1024, device=device) / (1024**0.5)

    # -------- LOGGING --------
   # raw_weight_logger = RawWeightLogger(f"../results/RFA_V2_4layer_raw_weights_{num_pdfs}.txt")
    stats_logger = WeightStatisticsLogger(f"../results/weights/RFA_V2_4layer_weight_stats_{num_pdfs}.txt")

    # -------- TRAINING LOGGING --------
    train_ce_hist, val_ce_hist = [], []
    train_acc_hist, val_acc_hist = [], []
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

                # -------- CE (LOGGING ONLY) --------
                ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()
                epoch_loss += ce.item()

                # -------- OUTPUT ERROR --------
                delta4 = (probs - y_onehot) / xb.size(0)

                # -------- RFA BACKWARD --------
                delta3 = (delta4 @ B4) * (a3 > 0).float()
                delta2 = (delta3 @ B3) * (a2 > 0).float()
                delta1 = (delta2 @ B2) * (a1 > 0).float()

                # -------- WEIGHT UPDATES --------
                model.fc4.weight -= lr * (delta4.T @ h3)
                model.fc3.weight -= lr * (delta3.T @ h2)
                model.fc2.weight -= lr * (delta2.T @ h1)
                model.fc1.weight -= lr * (delta1.T @ xb)

                model.fc4.bias -= lr * delta4.sum(dim=0)
                model.fc3.bias -= lr * delta3.sum(dim=0)
                model.fc2.bias -= lr * delta2.sum(dim=0)
                model.fc1.bias -= lr * delta1.sum(dim=0)

                # -------- LOG WEIGHTS --------
               # raw_weight_logger.log_batch(global_batch, model)
                stats_logger.log_batch(global_batch, model)

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
        val_acc_hist.append(val_acc)
        val_ce_hist.append(val_ce)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # Close loggers
   # raw_weight_logger.close()
    stats_logger.close()

    return max_acc_train, max_acc_val
