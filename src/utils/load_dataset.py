import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split


def prep_dataset(vector_size):

	PROJECT_ROOT = Path(f"/home1/vighnesh/Desktop/timit_rfa_dfa/data/processed_{vector_size}")
	print(PROJECT_ROOT)
	X = torch.load(PROJECT_ROOT/ "X.pt")
	Y = torch.load(PROJECT_ROOT/ "Y.pt")

	print(X.shape, Y.shape)

	dataset = TensorDataset(X,Y)
	val_ratio = 0.2
	val_size = int(len(dataset) * val_ratio)
	train_size = len(dataset) - val_size

	train_ds, val_ds = random_split(dataset,[train_size,val_size])

	train_loader = DataLoader(
		train_ds,
		batch_size = 256,
		shuffle = True,
		drop_last = True
	)

	val_loader = DataLoader(
		val_ds,
		batch_size = 256,
		shuffle = True,
		drop_last = True
	)


	return train_loader, val_loader
