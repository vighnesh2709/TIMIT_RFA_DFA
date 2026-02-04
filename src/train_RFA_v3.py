import torch
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from model.RFA import RFA_MLP


def train_rfa(X, Y, num_feats, num_pdfs):

    # ------------------ HYPERPARAMS (internal) ------------------
    epochs = 100
    batch_size = 256
    lr = 1e-3
    train_ratio = 0.9
    hidden_dim = 512
    
    # Profiling flag - set to True to enable profiling
    enable_profiling = True
    profile_epoch = 0  # Which epoch to profile (0 = first epoch)
    profile_batch_idx = 0  # Which batch in that epoch to profile (0 = first batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X = X.to(device)
    Y = Y.to(device)

    # ------------------ SPLIT ONCE ------------------
    N = X.size(0)
    perm = torch.randperm(N)

    train_N = int(train_ratio * N)
    train_idx = perm[:train_N]
    val_idx   = perm[train_N:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]

    # ------------------ MODEL ------------------
    model = RFA_MLP(num_feats, hidden_dim, num_pdfs).to(device)

    # Fixed random feedback matrices (RFA)
    B3 = torch.randn(num_pdfs, hidden_dim, device=device) / hidden_dim**0.5
    B2 = torch.randn(hidden_dim, hidden_dim, device=device) / hidden_dim**0.5

    # ------------------ LOGGING ------------------
    train_ce_hist, val_ce_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    max_acc_train = 0
    max_acc_val = 0
    
    # Profiling results storage
    profile_results = None

    # ------------------ TRAIN ------------------
    for epoch in range(epochs):

        perm = torch.randperm(X_train.size(0))
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        correct = 0
        total = 0
        epoch_loss = 0.0

        for batch_num, i in enumerate(tqdm(
            range(0, X_train.size(0), batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        )):
            xb = X_shuf[i:i+batch_size]
            yb = Y_shuf[i:i+batch_size]

            # ============================================================================
            # PROFILING: Run profiler on specified epoch and batch
            # ============================================================================
            if enable_profiling and epoch == profile_epoch and batch_num == profile_batch_idx:
                print("\n" + "="*80)
                print("PROFILING RFA FORWARD AND BACKWARD PASS")
                print("="*80)
                
                # Profile full training step (forward + RFA backward + updates)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA if device.type == "cuda" else torch.profiler.ProfilerActivity.CPU
                    ],
                    record_shapes=True,
                    with_flops=True,
                ) as prof:
                    with torch.no_grad():
                        # -------- forward --------
                        a1, h1, a2, h2, logits = model(xb)

                        logits = logits - logits.max(dim=1, keepdim=True).values
                        probs = torch.softmax(logits, dim=1)

                        # -------- one-hot --------
                        y_onehot = torch.zeros_like(probs)
                        y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                        # -------- CE (logging) --------
                        ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()

                        # -------- CE gradient --------
                        delta3 = (probs - y_onehot) / xb.size(0)

                        # -------- RFA backward --------
                        delta2 = (delta3 @ B3) * (a2 > 0).float()
                        delta1 = (delta2 @ B2) * (a1 > 0).float()

                        # -------- updates --------
                        model.fc3.weight -= lr * (delta3.T @ h2)
                        model.fc2.weight -= lr * (delta2.T @ h1)
                        model.fc1.weight -= lr * (delta1.T @ xb)

                        model.fc3.bias -= lr * delta3.sum(dim=0)
                        model.fc2.bias -= lr * delta2.sum(dim=0)
                        model.fc1.bias -= lr * delta1.sum(dim=0)

                # Extract profiling results
                prof_table = prof.key_averages()
                total_flops = sum([item.flops for item in prof_table if item.flops > 0])
                
                print(f"\nFull training step (Forward + RFA Backward + Updates):")
                print(f"  Total FLOPs: {total_flops:,}")
                print(f"  Total GFLOPs: {total_flops / 1e9:.6f}")
                print(f"  Batch size: {xb.size(0)}")
                print(f"  FLOPs per sample: {total_flops / xb.size(0):,}")
                
                print("\n" + "-"*80)
                print("TOP OPERATIONS BY FLOPs:")
                print("-"*80)
                print(prof_table.table(sort_by="flops", row_limit=15))
                
                # Store profiling results
                profile_results = {
                    "timestamp": datetime.now().isoformat(),
                    "device": str(device),
                    "epoch": profile_epoch,
                    "batch_number": profile_batch_idx,
                    "model_config": {
                        "num_features": num_feats,
                        "hidden_dim": hidden_dim,
                        "num_classes": num_pdfs,
                    },
                    "batch_config": {
                        "batch_size": xb.size(0),
                        "learning_rate": lr,
                    },
                    "flops": {
                        "full_training_step": {
                            "total_flops": int(total_flops),
                            "total_gflops": round(total_flops / 1e9, 6),
                            "per_sample": int(total_flops / xb.size(0)),
                        },
                    },
                    "operation_breakdown": _extract_op_breakdown(prof_table),
                }
                
                print("\n" + "="*80)
                print("Profiling complete. Results will be saved after training.\n")
                print("="*80 + "\n")
                
                # Continue with normal training after profiling
                with torch.no_grad():
                    # -------- forward --------
                    a1, h1, a2, h2, logits = model(xb)

                    logits = logits - logits.max(dim=1, keepdim=True).values
                    probs = torch.softmax(logits, dim=1)

                    # -------- one-hot --------
                    y_onehot = torch.zeros_like(probs)
                    y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                    # -------- CE (logging) --------
                    ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()
                    epoch_loss += ce.item()

                    # -------- CE gradient --------
                    delta3 = (probs - y_onehot) / xb.size(0)

                    # -------- RFA backward --------
                    delta2 = (delta3 @ B3) * (a2 > 0).float()
                    delta1 = (delta2 @ B2) * (a1 > 0).float()

                    # -------- updates --------
                    model.fc3.weight -= lr * (delta3.T @ h2)
                    model.fc2.weight -= lr * (delta2.T @ h1)
                    model.fc1.weight -= lr * (delta1.T @ xb)

                    model.fc3.bias -= lr * delta3.sum(dim=0)
                    model.fc2.bias -= lr * delta2.sum(dim=0)
                    model.fc1.bias -= lr * delta1.sum(dim=0)

                    # -------- accuracy --------
                    preds = probs.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            
            # ============================================================================
            # NORMAL TRAINING (No profiling)
            # ============================================================================
            else:
                with torch.no_grad():

                    # -------- forward --------
                    a1, h1, a2, h2, logits = model(xb)

                    logits = logits - logits.max(dim=1, keepdim=True).values
                    probs = torch.softmax(logits, dim=1)

                    # -------- one-hot --------
                    y_onehot = torch.zeros_like(probs)
                    y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                    # -------- CE (logging) --------
                    ce = -(y_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()
                    epoch_loss += ce.item()

                    # -------- CE gradient --------
                    delta3 = (probs - y_onehot) / xb.size(0)

                    # -------- RFA backward --------
                    delta2 = (delta3 @ B3) * (a2 > 0).float()
                    delta1 = (delta2 @ B2) * (a1 > 0).float()

                    # -------- updates --------
                    model.fc3.weight -= lr * (delta3.T @ h2)
                    model.fc2.weight -= lr * (delta2.T @ h1)
                    model.fc1.weight -= lr * (delta1.T @ xb)

                    model.fc3.bias -= lr * delta3.sum(dim=0)
                    model.fc2.bias -= lr * delta2.sum(dim=0)
                    model.fc1.bias -= lr * delta1.sum(dim=0)

                    # -------- accuracy --------
                    preds = probs.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

        train_acc = correct / total
        train_ce = epoch_loss / (X_train.size(0) / batch_size)
        max_acc_train = max(max_acc_train,train_acc)
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

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # ============================================================================
    # SAVE PROFILING RESULTS TO FILE (if profiling was enabled)
    # ============================================================================
    if enable_profiling and profile_results is not None:
        output_dir = Path("./rfa_profile_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / "rfa_flops_results.json"
        with open(json_path, "w") as f:
            json.dump(profile_results, f, indent=2)
        
        print("\n" + "="*80)
        print("PROFILING RESULTS SAVED")
        print("="*80)
        print(f"JSON file: {json_path}")
        
        # Also save a human-readable text file
        txt_path = output_dir / "rfa_flops_results.txt"
        _save_txt_summary(txt_path, profile_results)
        print(f"Text file: {txt_path}\n")

    return max_acc_train, max_acc_val


def _extract_op_breakdown(profiler_table):
    """Extract top operations by FLOPs from profiler table."""
    ops = []
    for item in profiler_table:
        if item.flops > 0:
            ops.append({
                "operation": item.key,
                "flops": int(item.flops),
                "cpu_time_ms": round(item.cpu_time / 1000, 4),
                "count": int(item.count),
            })
    
    ops.sort(key=lambda x: x["flops"], reverse=True)
    return ops[:15]


def _save_txt_summary(path, results):
    """Save human-readable summary to text file."""
    with open(path, "w") as f:
        f.write("="*80 + "\n")
        f.write("RFA TRAINING - FORWARD AND BACKWARD FLOPs PROFILING\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Device: {results['device']}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        for key, val in results['model_config'].items():
            f.write(f"  {key:.<40} {val}\n")
        
        f.write("\nBATCH CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        for key, val in results['batch_config'].items():
            f.write(f"  {key:.<40} {val}\n")
        
        f.write(f"\nProfiled at Epoch {results['epoch']}, Batch {results['batch_number']}\n\n")
        
        flops_data = results['flops']['full_training_step']
        f.write("FLOPs MEASUREMENT:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total FLOPs (Forward + RFA Backward + Updates):\n")
        f.write(f"  {flops_data['total_flops']:>20,} FLOPs\n")
        f.write(f"  {flops_data['total_gflops']:>20.6f} GFLOPs\n")
        f.write(f"  {flops_data['per_sample']:>20,} FLOPs per sample\n")
        
        f.write(f"\nEstimations:\n")
        f.write(f"  For 100 epochs (10K steps): {flops_data['total_flops'] * 10000:>15,} FLOPs\n")
        f.write(f"  For 100 epochs (10K steps): {flops_data['total_flops'] * 10000 / 1e12:>15.3f} TFLOPs\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP OPERATIONS BY FLOPs:\n")
        f.write("="*80 + "\n")
        f.write(f"{'Operation':<45} {'FLOPs':>15} {'Count':>8}\n")
        f.write("-"*80 + "\n")
        
        for op in results['operation_breakdown']:
            f.write(f"{op['operation']:<45} {op['flops']:>15,} {op['count']:>8}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NOTES:\n")
        f.write("="*80 + "\n")
        f.write("""
This profiling measures the FLOPs for one complete RFA training step which includes:
1. Forward pass through the network
2. RFA backward computation using fixed random matrices (B2, B3)
3. Direct weight updates

The FLOPs include all matrix multiplications and element-wise operations.

To scale this to a full training run:
- Multiply by number of total training steps (batches × epochs)
- Adjust batch size if different from profiled batch size
""")
