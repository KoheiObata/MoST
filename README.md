# MoST

This repository contains the official implementation for the paper []

## Requirements

The recommended requirements for MoST are specified as follows:
* Python 3.8.12
* torch==1.11
* scipy==1.6.1
* numpy==1.24.2
* pandas==2.0.0
* scikit_learn==0.24.2
* statsmodels==0.12.2
* Bottleneck==1.3.2

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [Google trend datasets] is put into `datasets/country` or `datasets/region` so that each data file can be located by `datasets/country/<ID>/<query>.csv`.
* [KnowAir datasets](https://github.com/shuowang-ai/PM2.5-GNN) should be put into `datasets/` so that each data file can be located by `datasets/KnowAir.npy`.

## Usage

To train and evaluate MoST on a dataset, run the following command:

```train & evaluate
python train.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --max-train-length <max_train_length> --repr-dims <repr_dims> --gpu <gpu> --epochs <epochs> --eval

python train.py e_commerce e_commerce --loader forecast_tensor --batch-size 8 --max-train-length 200 --repr-dims 320 --gpu 0 --epochs 100 --eval --seed 1
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `forecast_tensor` or `classification_tensor` or `encode_tensor`|
| batch_size | The batch size (defaults to 8) |
| max_train_length | The size of lookback window (defaults to 200) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/` and `result`.

**Scripts:** The scripts for reproduction are provided in `scripts/` folder.
