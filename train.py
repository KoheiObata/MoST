import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from most import MoST
import tasks
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

import utils_tensor

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to forecast_tensor or classification_tensor or encode_tensor')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate (defaults to 0.0001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--encode', action="store_true", help='Whether to encode data after training')

    parser.add_argument('--embedding', type=int, default=2, help='0:tok,pos,tem, 1:tok,pos, 2:tok,tem, 3:tok')
    parser.add_argument('--out_mode', type=str, default='mean', help='mean, linear, all_linear, max')
    parser.add_argument('--alpha', type=float, default=1, help='The ratio of dim loss')
    parser.add_argument('--hidden_dims', type=int, default=320, help='emmbedding dimension')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ')
    if args.loader == 'forecast_tensor':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, freq = utils_tensor.load_forecast_tensor(args.dataset, args)
        train_data = data[:, train_slice]

    elif args.loader == 'classification_tensor':
        task_type = 'classification'
        train_label, train_data, test_label, test_data, n_covariate_cols = utils_tensor.load_classification_tensor(args.dataset, args)
        freq = 'h'
        args.max_train_length = train_data.shape[1]
        args.embedding = 3

    elif args.loader == 'encode_tensor':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, freq = utils_tensor.load_encode_tensor(args.dataset)
        train_data = data[:, train_slice]

    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = MoST(
        input_dims=train_data.shape[-1],
        n_covariate_cols=n_covariate_cols,
        freq=freq,
        d1=train_data.shape[-2],
        task=task_type,
        out_mode=args.out_mode,
        alpha=args.alpha,
        embedding=args.embedding,
        device=device,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    model_t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=model_t)}\n")

    if args.eval:
        if task_type == 'forecasting':
            t = time.time()
            all_repr = model.encode(data, casual=True, sliding_length=1, sliding_padding=200, batch_size=4)
            MoST_infer_time = time.time() - t
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, all_repr, scaler, pred_lens, n_covariate_cols, MoST_infer_time, model_t, args)
            pkl_save(f'{run_dir}/out.pkl', out)
            pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
            print('Evaluation result:', eval_res)

        if task_type == 'classification':
            train_repr = model.encode(train_data, encoding_window='full_series' if train_label.ndim == 1 else None)
            t = time.time()
            test_repr = model.encode(test_data, encoding_window='full_series' if test_label.ndim == 1 else None)
            MoST_infer_time = time.time() - t
            out, eval_res = tasks.eval_classification(model, train_repr, train_label, test_repr, test_label,  args, MoST_infer_time, model_t, eval_protocol='linear')
            print('y_score',out)
            print('eval_res',eval_res)


    if args.encode:
        if args.dataset in ('knowairweek1', 'knowairweek2', 'knowairweek3'):
            batch_size=8
        else:
            batch_size=32
        padding=200
        all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=batch_size
        )
        print('encode_done')
        print(all_repr.shape)
        tasks.save_encode(all_repr, args)
        # np.save(f'{run_dir}/all_repr.npy', all_repr)

    print("Finished.")
