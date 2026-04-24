

CSV_PATH="$1"
OUTPUT_DIR="$2"



python minimol_ffn_binary.py \
  --train_csv train.csv \
  --val_csv val.csv \
  --test_csv test.csv \
  --cache_file cache.pkl \
  --batch_size 128 \
  --lr 1e-3 \
  --epochs 20 \
  --hidden_dim 512 \
  --num_layers 2 \
  --dropout 0.1