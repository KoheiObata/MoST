import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import time

import os
def make_dir(input_dir):
    if os.path.isdir(input_dir):
        print(f'{input_dir} already exist')
    else:
        os.makedirs(f"{input_dir}")
        print(f'{input_dir} is ready')

def eval_classification(model, train_repr, train_labels, test_repr, test_labels, args, infer_time=None, model_time=None, eval_protocol='linear', d=False):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    t = time.time()
    clf = fit_clf(train_repr, train_labels)
    fit_time = time.time() - t

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
        auprc = average_precision_score(test_labels_onehot, y_score)
    elif eval_protocol == 'svm':
        y_score = clf.decision_function(test_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
        auprc = average_precision_score(test_labels_onehot, y_score)
    else:
        y_score = 0
        auprc = 0

    save_dir='/opt/home/kohei/Contrastive_Tensor/src_baseline/MoST/result/'

    save_dir += f'/{args.dataset}'

    save_dir += f'/emb={args.embedding}'
    save_dir += f'/{args.out_mode}'
    save_dir += f'/alpha={args.alpha}'
    save_dir += f'/{args.max_train_length}'
    save_dir += f'/epoch={args.epochs}'
    save_dir += f'/eval={eval_protocol}'

    make_dir(save_dir)
    print('acc',eval_protocol,acc)
    np.savetxt(f'{save_dir}/{args.seed}.txt', np.array([acc]),fmt='%.4e')
    np.savetxt(f'{save_dir}/{args.seed}_modeltime.txt',np.array([model_time]),fmt='%.4e')
    np.savetxt(f'{save_dir}/{args.seed}_inftime.txt',np.array([infer_time]),fmt='%.4e')
    np.savetxt(f'{save_dir}/{args.seed}_fittime.txt',np.array([fit_time]),fmt='%.4e')
    return y_score, { 'acc': acc, 'auprc': auprc }
