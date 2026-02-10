from utils.process_timit import load_mfcc, load_alignments, check_dataset, write_tensor
from pathlib import Path
import torch
from utils.load_dataset import prep_dataset
from train_bp import train_bp
from train_bp_v2 import train_bp as train_bp_V2
from train_RFA import train_rfa
from train_RFA_v2 import train_rfa as train_rfa_V2
from train_DFA import train_dfa
from train_DFA_v2 import train_dfa as train_dfa_V2
import time
import os


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("TIMIT_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
    feature_dir = data_dir / "feature_extracted" / "export_feats"

    print("PRE PROCESSING 13 DIMENSIONAL VECTOR\n")

    load_mfcc(str(feature_dir / "mfcc_mono.txt"))
    load_alignments(str(feature_dir / "labels_mono.txt"))
    check_dataset()
    write_tensor("13")

    print("PRE PROCESSING 39 DIMENSIONAL VECTOR\n")

    load_mfcc(str(feature_dir / "mfcc_tri1.txt"))  # ✓ FIXED: Use feature_dir variable
    load_alignments(str(feature_dir / "labels_tri1.txt"))
    check_dataset()
    write_tensor("39")

    print("\n")

    # hmm-info mono/final.mdl 
    # number of phones 48
    # number of pdfs 144
    # number of transition-ids 288
    # number of transition-states 144

    splice_size = 5
    num_feats_13 = 13 * ((splice_size * 2) + 1)
    num_feats_39 = 39 * ((splice_size * 2) + 1)
    
    # =====================================================
    # 13 DIMENSIONS - TRAINING
    # =====================================================
    print("TRAINING BACKPROP 13 DIMENSIONS\n")
    train_loader, val_loader = prep_dataset("13", splice_size, True)
    
    start_bp = time.time()
    max_acc_train_bp_13, max_acc_val_bp_13 = train_bp(train_loader, val_loader, num_feats_13, 144)
    end_bp = time.time()
    
    print("BP Bigger model")
    start_bp_v2 = time.time()
    max_acc_train_bp_13_v2, max_acc_val_bp_13_v2 = train_bp_V2(train_loader, val_loader, num_feats_13, 144)
    end_bp_v2 = time.time()
    print("\n")
    
    print("TRAINING RFA 13 DIMENSIONS\n")
    processed_root = data_dir / "processed_13"
    X = torch.load(processed_root / "X.pt")
    Y = torch.load(processed_root / "Y.pt")
    
    start_rfa = time.time()
    max_acc_train_rfa_13, max_acc_val_rfa_13 = train_rfa(X, Y, num_feats_13, 144)
    end_rfa = time.time()
    
    print("RFA Bigger Model")
    start_rfa_v2 = time.time()
    max_acc_train_rfa_13_v2, max_acc_val_rfa_13_v2 = train_rfa_V2(X, Y, num_feats_13, 144)
    end_rfa_v2 = time.time()
    print("\n")

    print("TRAINING DFA 13 DIMENSIONS\n")
    start_dfa = time.time()
    max_acc_train_dfa_13, max_acc_val_dfa_13 = train_dfa(X, Y, num_feats_13, 144)
    end_dfa = time.time()
    
    print("DFA Bigger Model\n")
    start_dfa_v2 = time.time()
    max_acc_train_dfa_13_v2, max_acc_val_dfa_13_v2 = train_dfa_V2(X, Y, num_feats_13, 144)
    end_dfa_v2 = time.time()

    # =====================================================
    # 39 DIMENSIONS - TRAINING
    # =====================================================
    # hmm-info tri1/final.mdl 
    # number of phones 48
    # number of pdfs 1880
    # number of transition-ids 3834
    # number of transition-states 1917

    print("\n")
    
    print("TRAINING BACKPROP 39 DIMENSIONS\n")
    train_loader, val_loader = prep_dataset("39", splice_size, True)
    
    start_bp_39 = time.time()
    max_acc_train_bp_39, max_acc_val_bp_39 = train_bp(train_loader, val_loader, num_feats_39, 1880)
    end_bp_39 = time.time()
    
    print("BP Bigger model")
    start_bp_39_v2 = time.time()
    max_acc_train_bp_39_v2, max_acc_val_bp_39_v2 = train_bp_V2(train_loader, val_loader, num_feats_39, 1880)
    end_bp_39_v2 = time.time()
    print("\n")
    
    print("TRAINING RFA 39 DIMENSIONS\n")
    processed_root = data_dir / "processed_39"
    X = torch.load(processed_root / "X.pt")
    Y = torch.load(processed_root / "Y.pt")
    
    start_rfa_39 = time.time()
    max_acc_train_rfa_39, max_acc_val_rfa_39 = train_rfa(X, Y, num_feats_39, 1880)
    end_rfa_39 = time.time()

    print("RFA Bigger Model")    
    start_rfa_39_v2 = time.time()
    max_acc_train_rfa_39_v2, max_acc_val_rfa_39_v2 = train_rfa_V2(X, Y, num_feats_39, 1880)
    end_rfa_39_v2 = time.time()
    print("\n")

    print("TRAINING DFA 39 DIMENSIONS\n")
    start_dfa_39 = time.time()
    max_acc_train_dfa_39, max_acc_val_dfa_39 = train_dfa(X, Y, num_feats_39, 1880)
    end_dfa_39 = time.time()

    print("DFA Bigger Model")
    start_dfa_39_v2 = time.time()
    max_acc_train_dfa_39_v2, max_acc_val_dfa_39_v2 = train_dfa_V2(X, Y, num_feats_39, 1880)
    end_dfa_39_v2 = time.time()
    
    # =====================================================
    # WRITE RESULTS TO FILE
    # =====================================================
    with open("../results/results_comparison.txt", "w") as write_file:
        
        write_file.write("\n" + "="*100 + "\n")
        write_file.write("SUMMARY 13 DIMENSIONS (144 OUTPUT CLASSES)\n")
        write_file.write("="*100 + "\n")
        
        write_file.write("Standard Model:\n")
        write_file.write(f"  Backprop  | Train Acc: {max_acc_train_bp_13:.4f} | Val Acc: {max_acc_val_bp_13:.4f} | Time: {(end_bp - start_bp)/60:.2f} min\n")
        write_file.write(f"  RFA       | Train Acc: {max_acc_train_rfa_13:.4f} | Val Acc: {max_acc_val_rfa_13:.4f} | Time: {(end_rfa - start_rfa)/60:.2f} min\n")
        write_file.write(f"  DFA       | Train Acc: {max_acc_train_dfa_13:.4f} | Val Acc: {max_acc_val_dfa_13:.4f} | Time: {(end_dfa - start_dfa)/60:.2f} min\n")
        
        write_file.write("\nLarger Model (V2):\n")
        write_file.write(f"  Backprop  | Train Acc: {max_acc_train_bp_13_v2:.4f} | Val Acc: {max_acc_val_bp_13_v2:.4f} | Time: {(end_bp_v2 - start_bp_v2)/60:.2f} min\n")
        write_file.write(f"  RFA       | Train Acc: {max_acc_train_rfa_13_v2:.4f} | Val Acc: {max_acc_val_rfa_13_v2:.4f} | Time: {(end_rfa_v2 - start_rfa_v2)/60:.2f} min\n")
        write_file.write(f"  DFA       | Train Acc: {max_acc_train_dfa_13_v2:.4f} | Val Acc: {max_acc_val_dfa_13_v2:.4f} | Time: {(end_dfa_v2 - start_dfa_v2)/60:.2f} min\n")
        
        write_file.write("\n" + "="*100 + "\n")
        write_file.write("SUMMARY 39 DIMENSIONS (1880 OUTPUT CLASSES)\n")
        write_file.write("="*100 + "\n")
        
        write_file.write("Standard Model:\n")
        write_file.write(f"  Backprop  | Train Acc: {max_acc_train_bp_39:.4f} | Val Acc: {max_acc_val_bp_39:.4f} | Time: {(end_bp_39 - start_bp_39)/60:.2f} min\n")
        write_file.write(f"  RFA       | Train Acc: {max_acc_train_rfa_39:.4f} | Val Acc: {max_acc_val_rfa_39:.4f} | Time: {(end_rfa_39 - start_rfa_39)/60:.2f} min\n")
        write_file.write(f"  DFA       | Train Acc: {max_acc_train_dfa_39:.4f} | Val Acc: {max_acc_val_dfa_39:.4f} | Time: {(end_dfa_39 - start_dfa_39)/60:.2f} min\n")
        
        write_file.write("\nLarger Model (V2):\n")
        write_file.write(f"  Backprop  | Train Acc: {max_acc_train_bp_39_v2:.4f} | Val Acc: {max_acc_val_bp_39_v2:.4f} | Time: {(end_bp_39_v2 - start_bp_39_v2)/60:.2f} min\n")
        write_file.write(f"  RFA       | Train Acc: {max_acc_train_rfa_39_v2:.4f} | Val Acc: {max_acc_val_rfa_39_v2:.4f} | Time: {(end_rfa_39_v2 - start_rfa_39_v2)/60:.2f} min\n")
        write_file.write(f"  DFA       | Train Acc: {max_acc_train_dfa_39_v2:.4f} | Val Acc: {max_acc_val_dfa_39_v2:.4f} | Time: {(end_dfa_39_v2 - start_dfa_39_v2)/60:.2f} min\n")
        
        write_file.write("\n" + "="*100 + "\n")
    
    print("\nResults saved to ../results/results_comparison.txt")


if __name__ == "__main__":
    main()
