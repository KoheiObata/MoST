gpu=2
seed=0
python train.py synthetic_20_20_3_3 synthetic_20_20_3_3 --loader encode_tensor --batch-size 8 --max-train-length 200 --repr-dims 320 --gpu $gpu --epochs 1 --encode --seed $seed --out_mode mean --embedding 2