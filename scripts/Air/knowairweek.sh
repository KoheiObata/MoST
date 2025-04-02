gpu=0
name=knowairweek1
for seed in $(seq 0 4); do
	python train.py $name $name --loader forecast_tensor --batch-size 8 --max-train-length 200 --repr-dims 320 --gpu $gpu --epochs 100 --eval --seed ${seed} --out_mode mean --alpha 1
done

name=knowairweek2
for seed in $(seq 0 4); do
	python train.py $name $name --loader forecast_tensor --batch-size 8 --max-train-length 200 --repr-dims 320 --gpu $gpu --epochs 100 --eval --seed ${seed} --out_mode mean --alpha 1
done

name=knowairweek3
for seed in $(seq 0 4); do
	python train.py $name $name --loader forecast_tensor --batch-size 8 --max-train-length 200 --repr-dims 320 --gpu $gpu --epochs 100 --eval --seed ${seed} --out_mode mean --alpha 1
done