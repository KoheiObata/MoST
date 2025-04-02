import numpy as np

import os
def make_dir(input_dir):
    if os.path.isdir(input_dir):
        print(f'{input_dir} already exist')
    else:
        os.makedirs(f"{input_dir}")
        print(f'{input_dir} is ready')

def save_encode(repr, args):

    save_dir='/opt/home/kohei/Contrastive_Tensor/src_baseline/MoST/result/'

    save_dir += f'/{args.dataset}'

    save_dir += f'/emb={args.embedding}'
    save_dir += f'/{args.out_mode}'
    save_dir += f'/alpha={args.alpha}'
    save_dir += f'/epoch={args.epochs}'
    save_dir += f'/{args.max_train_length}'

    make_dir(save_dir)
    np.save(f'{save_dir}/all_repr.npy', repr)
