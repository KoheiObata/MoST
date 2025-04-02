import numpy as np
import time
import os
from . import _eval_protocols as eval_protocols


def make_dir(input_dir):
    if os.path.isdir(input_dir):
        print(f'{input_dir} already exist')
    else:
        os.makedirs(f"{input_dir}")
        print(f'{input_dir} is ready')

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }

def nan_to_zero(data):
    mask = np.isnan(data)
    data[mask] = 0
    return data, mask

def eval_forecasting(model, data, train_slice, valid_slice, test_slice, all_repr, scaler, pred_lens, n_covariate_cols, MoST_infer_time, model_time, args, d=False):
    padding = 200


    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    train_data = data[:, train_slice, :, n_covariate_cols:]
    valid_data = data[:, valid_slice, :, n_covariate_cols:]
    test_data = data[:, test_slice, :, n_covariate_cols:]

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2]*train_data.shape[3])
    valid_data = valid_data.reshape(valid_data.shape[0], valid_data.shape[1], valid_data.shape[2]*valid_data.shape[3])
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2]*test_data.shape[3])

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    valid_result, alpha_result = [], []
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        train_labels, train_mask = nan_to_zero(train_labels)
        valid_labels, valid_mask = nan_to_zero(valid_labels)
        test_labels, test_mask = nan_to_zero(test_labels)

        t = time.time()
        lr, valid_score, best_alpha = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels, train_mask, valid_mask)
        valid_result.append(valid_score)
        alpha_result.append(best_alpha)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2] #dim, time, pred_len, pred_dim
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        out_log[pred_len] = {
            'norm': test_pred,
            'norm_gt': test_labels,
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
        }

    eval_res = {
        'ours': ours_result,
        'MoST_infer_time': MoST_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    result_mse=[]
    result_mae=[]
    for pred_len in pred_lens:
        result_mse.append(ours_result[pred_len]['norm']['MSE'])
        result_mae.append(ours_result[pred_len]['norm']['MAE'])
    result=np.concatenate((np.array(result_mse)[np.newaxis, :], np.array(result_mae)[np.newaxis, :]), axis=0)
    validalpha_result=np.concatenate((np.array(valid_result)[np.newaxis, :], np.array(alpha_result)[np.newaxis, :]), axis=0)

    result_tr=[]
    result_in=[]
    for pred_len in pred_lens:
        result_tr.append(lr_train_time[pred_len])
        result_in.append(lr_infer_time[pred_len])
    result_time=np.concatenate((np.array(result_tr)[np.newaxis, :], np.array(result_in)[np.newaxis, :]), axis=0)

    save_dir='/opt/home/kohei/Contrastive_Tensor/src_baseline/MoST/result/'
    save_dir += f'/{args.dataset}'

    save_dir += f'/emb={args.embedding}'
    save_dir += f'/{args.out_mode}'
    save_dir += f'/alpha={args.alpha}'
    save_dir += f'/epoch={args.epochs}'
    save_dir += f'/{args.max_train_length}'

    make_dir(save_dir)
    np.savetxt(f'{save_dir}/{args.seed}.txt',result.T,fmt='%.4e')
    np.savetxt(f'{save_dir}/{args.seed}_validalpha.txt',validalpha_result.T,fmt='%.4e')
    np.savetxt(f'{save_dir}/{args.seed}time.txt',result_time.T,fmt='%.4e')
    np.savetxt(f'{save_dir}/{args.seed}_modeltime.txt',np.array([model_time]),fmt='%.4e')
    np.savetxt(f'{save_dir}/{args.seed}_inftime.txt',np.array([MoST_infer_time]),fmt='%.4e')
    return out_log, eval_res
