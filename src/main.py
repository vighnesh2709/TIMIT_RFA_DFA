from utils.process_timit import load_mfcc, load_alignments, check_dataset,write_tensor
from pathlib import Path
import torch
from utils.load_dataset import prep_dataset
from train_bp import train_bp
from train_RFA import train_rfa
from train_DFA import train_dfa
import time

def main():


	print("PRE PROCESSING 13 DIMENTIONAL VECTOR\n")

	load_mfcc("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/mfcc_mono.txt")
	load_alignments("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/labels_mono.txt")
	check_dataset()
	write_tensor("13")

	print("PRE PROCESSING 39 DIMENTIONAL VECTOR\n")

	load_mfcc("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/mfcc_tri1.txt")
	load_alignments("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/labels_tri1.txt")
	check_dataset()
	write_tensor("39")

	print("\n")

	# hmm-info mono/final.mdl 
	# number of phones 48
	# number of pdfs 144
	# number of transition-ids 288
	# number of transition-states 144

	splice_size = 5
	
	print("TRAINING BACKPROP 13 DIMENTIONS\n")
	train_loader, val_loader = prep_dataset("13", splice_size, True)
	start_bp = time.time()
	max_acc_train_bp_13,max_acc_val_bp_13 = train_bp(train_loader, val_loader,13*((splice_size*2)+1),144)
	end_bp = time.time()
	print("\n")
	
	print("TRAINING RFA\n")
	PROJECT_ROOT = Path("/home1/vighnesh/Desktop/timit_rfa_dfa/data/processed_13")
	X = torch.load(PROJECT_ROOT/ "X.pt")
	Y = torch.load(PROJECT_ROOT/ "Y.pt")
	start_RFA = time.time()
	max_acc_train_RFA_13,max_acc_val_RFA_13 = train_rfa(X,Y,13*((splice_size*2)+1),144)
	end_RFA = time.time()

	print("\n")

	print("TRAINING DFA\n")
	start_DFA = time.time()
	max_acc_train_DFA_13,max_acc_val_DFA_13 = train_dfa(X,Y,13*((splice_size*2)+1),144)
	end_DFA = time.time()

	# hmm-info tri1/final.mdl 
	# number of phones 48
	# number of pdfs 1880
	# number of transition-ids 3834
	# number of transition-states 1917

	print("\n")
	
	print("TRAINING BACKPROP 39 DIMENTIONS\n")
	train_loader, val_loader = prep_dataset("39",splice_size, True)
	start_bp_39 = time.time()
	max_acc_train_bp_39, max_acc_val_bp_39 = train_bp(train_loader, val_loader,39*((splice_size*2)+1),1880)
	end_bp_39 = time.time()
	print("\n")
	
	print("TRAINING RFA\n")
	PROJECT_ROOT = Path("/home1/vighnesh/Desktop/timit_rfa_dfa/data/processed_39")
	X = torch.load(PROJECT_ROOT/ "X.pt")
	Y = torch.load(PROJECT_ROOT/ "Y.pt")
	start_RFA_39 = time.time()
	max_acc_train_RFA_39, max_acc_val_RFA_39 = train_rfa(X,Y,39*((splice_size*2)+1),1880)
	end_RFA_39 = time.time()

	# print("\n")

	print("TRAINING DFA\n")
	start_DFA_39 = time.time()
	max_acc_train_DFA_39, max_acc_val_DFA_39 = train_dfa(X,Y,39*((splice_size*2)+1),1880)
	end_DFA_39 = time.time()


	print("\n================ SUMMARY 13 DIMENSIONS =================")
	print(f"Backprop  | Max Train Acc: {max_acc_train_bp_13:.4f} | Max Val Acc: {max_acc_val_bp_13:.4f} | Time: {(end_bp - start_bp)/60:.2f} min")
	print(f"RFA       | Max Train Acc: {max_acc_train_RFA_13:.4f} | Max Val Acc: {max_acc_val_RFA_13:.4f} | Time: {(end_RFA - start_RFA)/60:.2f} min")
	print(f"DFA       | Max Train Acc: {max_acc_train_DFA_13:.4f} | Max Val Acc: {max_acc_val_DFA_13:.4f} | Time: {(end_DFA - start_DFA)/60:.2f} min")
	print("========================================================")

	print("\n================ SUMMARY 39 DIMENSIONS =================")
	print(f"Backprop  | Max Train Acc: {max_acc_train_bp_39:.4f} | Max Val Acc: {max_acc_val_bp_39:.4f} | Time: {(end_bp_39 - start_bp_39)/60:.2f} min")
	print(f"RFA       | Max Train Acc: {max_acc_train_RFA_39:.4f} | Max Val Acc: {max_acc_val_RFA_39:.4f} | Time: {(end_RFA_39 - start_RFA_39)/60:.2f} min")
	print(f"DFA       | Max Train Acc: {max_acc_train_DFA_39:.4f} | Max Val Acc: {max_acc_val_DFA_39:.4f} | Time: {(end_DFA_39 - start_DFA_39)/60:.2f} min")
	print("========================================================")



	

if __name__ ==  "__main__":
	main()
