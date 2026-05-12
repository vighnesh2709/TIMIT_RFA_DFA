import torch
from pathlib import Path
import re


mapping = {}
mfcc = {}
transition = {}
phone_id = {}
alignments = {}



PROJECT_ROOT = Path(__file__).resolve().parents[2]

# def load_alignments(path):
#     with open(path, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             utt_id = parts[0]
#             alignments[utt_id] = [int(x) for x in parts[1:]]


def load_alignments(path,transition_path):
    pattern = re.compile(r'phone\s*=\s*(\w+)\s+hmm-state\s*=\s*(\d+)\s+pdf\s*=\s*(\d+)')
    count = 0
    with open(transition_path) as f:
        for line in f:
            m = pattern.search(line)
            phone = m.group(1)
            pdf = int(m.group(3))

            if phone not in phone_id:
                phone_id[phone] = count
                count += 1

            pid = phone_id[phone]

            if pdf not in mapping:
                mapping[pdf] = pid
            
            
    print(len(mapping))
    print(mapping)

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[0]
            alignments[utt_id] = []
            for vals in parts[1:]:
                # print(mapping[int(vals)])
                alignments[utt_id].append(mapping[int(vals)])
            # alignments[utt_id] = [int(x) for x in parts[1:]]



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
    if "FRLL0_SI805" in mfcc:
        del mfcc["FRLL0_SI805"]
    if "FRLL0_SI805" in alignments:
        del alignments["FRLL0_SI805"]

    assert set(mfcc.keys()) == set(alignments.keys())

    for utt in mfcc:
        assert len(mfcc[utt]) == len(alignments[utt])

    print("Frame label alignments are correct")


def write_tensor(vector_size):
    X = []
    Y = []

    out_dir = PROJECT_ROOT / "data" / f"processed_{vector_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "processed_data.txt"

    with open(txt_path, "w") as file:

        for utt in mfcc:
            for frame, label in zip(mfcc[utt], alignments[utt]):
                X.append(frame)
                Y.append(label)

                file.write(" ".join(map(str, frame)) + f" {label}\n")

    print(f"Number of classes: {len(set(Y))}")
    print(f"Size X: {len(X)}")
    print(f"Size Y: {len(Y)}")

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.int64)

    torch.save(X, out_dir / "X.pt")
    torch.save(Y, out_dir / "Y.pt")
