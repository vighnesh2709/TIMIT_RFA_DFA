import os
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split

def splice_data(X,context):
	T,D = X.shape
	X_ctx = []

	for t in range(T):
		frames = []
		for offset in range(-context,context+1):
			idx = t + offset
			if idx < 0:
				idx = 0
			elif idx >= T:
				idx = T-1
			frames.append(X[idx])
		X_ctx.append(torch.cat(frames))

	return torch.stack(X_ctx)


def prep_dataset(vector_size,context = 5,splicing = False):

	# PROJECT_ROOT = Path(f"/home1/vighnesh/Desktop/timit_rfa_dfa/data/processed_{vector_size}")
	# print(PROJECT_ROOT)
	# X = torch.load(PROJECT_ROOT/ "X.pt")
	# Y = torch.load(PROJECT_ROOT/ "Y.pt")

	project_root = Path(__file__).resolve().parents[2]
	data_dir = Path(os.environ.get("TIMIT_DATA_DIR", project_root / "data")).expanduser()
	processed_root = data_dir / f"processed_{vector_size}"
	print(processed_root)
	X = torch.load(processed_root / "X.pt")
	Y = torch.load(processed_root / "Y.pt")

	# print(X.shape, Y.shape)
	
	if splicing:

		X = splice_data(X,context)

		# torch.save(X, PROJECT_ROOT / "X.pt")
		# torch.save(Y, PROJECT_ROOT / "Y.pt")
		torch.save(X, processed_root / "X.pt")
		torch.save(Y, processed_root / "Y.pt")

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
