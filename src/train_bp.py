import torch
import torch.nn as nn
from tqdm import tqdm
from model.backprop import MLP
from model.backprop_v2 import MLP as MLP_V2


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


def train_bp(train_loader, val_loader, num_feats, num_pdfs):
    
    # -------- MODEL --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = MLP(num_feats, num_pdfs).to(device)
    no_param = 0
    for p in model.parameters():
        no_param += len(p)
    print(no_param)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # -------- LOGGING --------
   # raw_weight_logger = RawWeightLogger(f"../results/BP_raw_weights_{num_pdfs}.txt")
    stats_logger = WeightStatisticsLogger(f"../results/weights/BP_weight_stats_{num_pdfs}.txt")
    
    # -------- TRAIN --------
    epochs = 20
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    max_acc_train = 0
    max_acc_val = 0
    global_batch = 0
    
    for epoch in range(epochs):
        
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
            
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # -------- LOG WEIGHTS --------
           # raw_weight_logger.log_batch(global_batch, model)
            stats_logger.log_batch(global_batch, model)
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            global_batch += 1
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        max_acc_train = max(max_acc_train, train_acc)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # -------- VALIDATION --------
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
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
    
    # Close loggers
    #raw_weight_logger.close()
    stats_logger.close()
    
    return max_acc_train, max_acc_val
