gpu=2
for seed in $(seq 0 4); do
	python train.py daily/seed=$seed daily/seed=$seed --loader classification_tensor --batch-size 8 --max-train-length 200 --repr-dims 320 --epochs 20 --gpu $gpu --eval --seed ${seed} --out_mode max --alpha 10
done
for seed in $(seq 0 4); do
	python train.py realdisp/seed=$seed realdisp/seed=$seed --loader classification_tensor --batch-size 8 --max-train-length 200 --repr-dims 320 --epochs 20 --gpu $gpu --eval --seed ${seed} --out_mode max --alpha 10
done