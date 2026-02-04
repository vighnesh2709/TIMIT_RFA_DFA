import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from model.backprop import MLP 
from model.backprop_v2 import MLP as MLP_V2


def train_bp(train_loader, val_loader, num_feats, num_pdfs):
    # ---------- MODEL ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = MLP_V2(num_feats, num_pdfs).to(device)
    no_param = 0
    for p in model.parameters():
        no_param += len(p)
    print(no_param)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # ---------- PROFILING CONFIG ----------
    enable_profiling = True
    profile_epoch = 0  # Which epoch to profile (0 = first epoch)
    profile_batch_idx = 0  # Which batch in that epoch to profile (0 = first batch)
    
    # ---------- TRAIN ----------
    epochs = 100
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    max_acc_train = 0
    max_acc_val = 0
    profile_results = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)):
            x = x.to(device)
            y = y.to(device)
            
            # ============================================================================
            # PROFILING: Run profiler on specified epoch and batch
            # ============================================================================
            if enable_profiling and epoch == profile_epoch and batch_idx == profile_batch_idx:
                print("\n" + "="*80)
                print("PROFILING STANDARD BACKPROP (FORWARD + BACKWARD + OPTIMIZER STEP)")
                print("="*80)
                
                # Profile full training step (forward + backward + optimizer.step())
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA if device.type == "cuda" else torch.profiler.ProfilerActivity.CPU
                    ],
                    record_shapes=True,
                    with_flops=True,
                ) as prof:
                    # Forward pass
                    logits = model(x)
                    loss = criterion(logits, y)
                    
                    # Backward pass (this is where PyTorch computes gradients)
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Optimizer step (weight updates)
                    optimizer.step()
                
                # Extract profiling results
                prof_table = prof.key_averages()
                total_flops = sum([item.flops for item in prof_table if item.flops > 0])
                
                print(f"\nFull training step (Forward + Backward + Optimizer step):")
                print(f"  Total FLOPs: {total_flops:,}")
                print(f"  Total GFLOPs: {total_flops / 1e9:.6f}")
                print(f"  Batch size: {x.size(0)}")
                print(f"  FLOPs per sample: {total_flops / x.size(0):,}")
                
                print("\n" + "-"*80)
                print("TOP OPERATIONS BY FLOPs:")
                print("-"*80)
                print(prof_table.table(sort_by="flops", row_limit=20))
                
                # Store profiling results
                profile_results = {
                    "timestamp": datetime.now().isoformat(),
                    "device": str(device),
                    "method": "Standard Backpropagation (PyTorch)",
                    "epoch": profile_epoch,
                    "batch_number": profile_batch_idx,
                    "model_config": {
                        "num_features": num_feats,
                        "num_classes": num_pdfs,
                        "total_parameters": no_param,
                    },
                    "batch_config": {
                        "batch_size": x.size(0),
                        "learning_rate": 1e-3,
                        "optimizer": "SGD",
                    },
                    "flops": {
                        "full_training_step": {
                            "total_flops": int(total_flops),
                            "total_gflops": round(total_flops / 1e9, 6),
                            "per_sample": int(total_flops / x.size(0)),
                        },
                    },
                    "operation_breakdown": _extract_op_breakdown(prof_table),
                }
                
                print("\n" + "="*80)
                print("Profiling complete. Results will be saved after training.\n")
                print("="*80 + "\n")
            
            # ============================================================================
            # NORMAL TRAINING (No profiling)
            # ============================================================================
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
        max_acc_train = max(max_acc_train, train_acc)
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
    
    # ============================================================================
    # SAVE PROFILING RESULTS TO FILE (if profiling was enabled)
    # ============================================================================
    if enable_profiling and profile_results is not None:
        output_dir = Path("./backprop_profile_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / "backprop_flops_results.json"
        with open(json_path, "w") as f:
            json.dump(profile_results, f, indent=2)
        
        print("\n" + "="*80)
        print("PROFILING RESULTS SAVED")
        print("="*80)
        print(f"JSON file: {json_path}")
        
        # Also save a human-readable text file
        txt_path = output_dir / "backprop_flops_results.txt"
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
    return ops[:20]


def _save_txt_summary(path, results):
    """Save human-readable summary to text file."""
    with open(path, "w") as f:
        f.write("="*80 + "\n")
        f.write("STANDARD BACKPROPAGATION - FLOPs PROFILING\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Device: {results['device']}\n")
        f.write(f"Method: {results['method']}\n\n")
        
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
        f.write(f"Total FLOPs (Forward + Backward + Optimizer):\n")
        f.write(f"  {flops_data['total_flops']:>20,} FLOPs\n")
        f.write(f"  {flops_data['total_gflops']:>20.6f} GFLOPs\n")
        f.write(f"  {flops_data['per_sample']:>20,} FLOPs per sample\n")
        
        f.write(f"\nEstimations:\n")
        # Assuming ~100 epochs, estimate batches per epoch
        f.write(f"  For 100 epochs training: multiply by total batches\n")
        f.write(f"  Example: {flops_data['total_flops']} × 10,000 batches\n")
        f.write(f"          = {flops_data['total_flops'] * 10000:,} FLOPs\n")
        f.write(f"          = {flops_data['total_flops'] * 10000 / 1e12:.3f} TFLOPs\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP OPERATIONS BY FLOPs:\n")
        f.write("="*80 + "\n")
        f.write(f"{'Operation':<45} {'FLOPs':>15} {'Count':>8}\n")
        f.write("-"*80 + "\n")
        
        for op in results['operation_breakdown']:
            f.write(f"{op['operation']:<45} {op['flops']:>15,} {op['count']:>8}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY INSIGHT:\n")
        f.write("="*80 + "\n")
        f.write("""
This profiling measures the FLOPs for one complete training step which includes:

1. Forward pass: Model inference (x → logits)
2. Backward pass: PyTorch autograd computes gradients for all parameters
3. Optimizer step: SGD applies gradient updates to weights

The backward pass dominates and is typically 2-3x more expensive than forward.

Compare this with RFA and DFA methods to see compute savings.
""")
