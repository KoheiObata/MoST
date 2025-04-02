gpu=0
name=e_commerce
for seed in $(seq 0 0); do
	python train.py $name $name --loader forecast_tensor --batch-size 8 --max-train-length 200 --repr-dims 10 --gpu $gpu --epochs 1 --eval --seed ${seed}
done
name=vod
name=sweets
name=facilities
name=music
name=SNS
name=apparel