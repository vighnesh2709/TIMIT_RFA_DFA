import torch
from pathlib import Path


alignments = {}
mfcc = {}



PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_alignments(path):
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[0]
            alignments[utt_id] = [int(x) for x in parts[1:]]


def load_mfcc(path):
    global mfcc
    mfcc = {}

    current_utt = None
    frames = []

    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue

            if len(tokens) == 2 and tokens[1] == "[":
                current_utt = tokens[0]
                frames = []
                continue

            if tokens == ["]"]:
                mfcc[current_utt] = frames
                current_utt = None
                continue

            if tokens[-1] == "]":
                frames.append([float(x) for x in tokens[:-1]])
                mfcc[current_utt] = frames
                current_utt = None
                continue

            frames.append([float(x) for x in tokens])

    if current_utt is not None:
        mfcc[current_utt] = frames



def check_dataset():
    assert set(mfcc.keys()) == set(alignments.keys())

    for utt in mfcc:
        assert len(mfcc[utt]) == len(alignments[utt])

    print("Frame label alignments are correct")


def write_tensor(vector_size):
	X = []
	Y = []

	for utt in mfcc:
		for f,y in zip(mfcc[utt],alignments[utt]):
			X.append(f)
			Y.append(y)

	X = torch.tensor(X,dtype = torch.float32)
	Y = torch.tensor(Y,dtype = torch.int64)

	out_dir = PROJECT_ROOT / "data" / f"processed_{vector_size}"
	out_dir.mkdir(parents=True, exist_ok=True)
	
	torch.save(X, out_dir / "X.pt")
	torch.save(Y, out_dir / "Y.pt")
