from utils.process_timit import load_mfcc, load_alignments, check_dataset,write_tensor
from pathlib import Path
import torch
from utils.load_dataset import prep_dataset
from train_bp import train_bp
from train_RFA import train_rfa

def main():
	load_mfcc("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/mfcc_mono.txt")
	load_alignments("/home1/vighnesh/Desktop/timit_rfa_dfa/data/feature_extracted/export_feats/labels_mono.txt")
	check_dataset()
	write_tensor()
	train_loader, val_loader = prep_dataset()
	print(train_loader, val_loader)
	train_bp(train_loader, val_loader,13,144)
	
	PROJECT_ROOT = Path("/home1/vighnesh/Desktop/timit_rfa_dfa/data/processed")
	X = torch.load(PROJECT_ROOT/ "X.pt")
	Y = torch.load(PROJECT_ROOT/ "Y.pt")
	train_rfa(X,Y,13,144)
	

	
if __name__ ==  "__main__":
	main()
