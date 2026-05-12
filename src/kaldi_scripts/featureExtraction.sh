set -e
source ./path.sh

OUT_DIR=export_feats
mkdir -p $OUT_DIR

apply-cmvn --utt2spk=ark:data/train/utt2spk \
  scp:data/train/cmvn.scp scp:data/train/feats.scp \
  ark,t:$OUT_DIR/mfcc_mono.txt

apply-cmvn --utt2spk=ark:data/train/utt2spk \
  scp:data/train/cmvn.scp scp:data/train/feats.scp ark:- | \
add-deltas ark:- ark,t:$OUT_DIR/mfcc_tri1.txt

ali-to-pdf exp/mono/final.mdl "ark:gunzip -c exp/mono_ali/ali.*.gz|" \
  ark,t:$OUT_DIR/labels_mono.txt

ali-to-pdf exp/tri1/final.mdl "ark:gunzip -c exp/tri1_ali/ali.*.gz|" \
  ark,t:$OUT_DIR/labels_tri1.txt

echo "Feature and Labels exported"
