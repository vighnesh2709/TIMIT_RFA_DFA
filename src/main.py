from utils.process_timit import load_mfcc, load_alignments, check_dataset,write_tensor
from pathlib import Path
import torch
from utils.load_dataset import prep_dataset
from train_bp import train_bp
from train_RFA import train_rfa
from train_DFA import train_dfa
import time

def main():


	print("PRE PROCESSING\n")

	load_mfcc("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/mfcc_mono.txt")
	load_alignments("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/labels_mono.txt")
	check_dataset()
	write_tensor()

	print("\n")
	
	print("TRAINING BACKPROP\n")
	train_loader, val_loader = prep_dataset()
	start_bp = time.time()
	train_bp(train_loader, val_loader,13,144)
	end_bp = time.time()
	print("\n")
	
	print("TRAINING RFA\n")
	PROJECT_ROOT = Path("/home1/vighnesh/Desktop/timit_rfa_dfa/data/processed")
	X = torch.load(PROJECT_ROOT/ "X.pt")
	Y = torch.load(PROJECT_ROOT/ "Y.pt")
	start_RFA = time.time()
	train_rfa(X,Y,13,144)
	end_RFA = time.time()

	print("\n")

	print("TRAINING DFA\n")
	start_DFA = time.time()
	train_dfa(X,Y,13,144)
	end_DFA = time.time()

	print("\n================ TIME SUMMARY ================")
	print(f"Backprop Training Time : {(end_bp - start_bp)/60:.2f} minutes")
	print(f"RFA Training Time      : {(end_RFA - start_RFA)/60:.2f} minutes")
	print(f"DFA Training Time      : {(end_DFA - start_DFA)/60:.2f} minutes")
	print("==============================================")




if __name__ ==  "__main__":
	main()
