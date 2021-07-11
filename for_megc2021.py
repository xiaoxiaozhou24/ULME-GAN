import os
import sys
dirpaths = os.listdir('test')
for dirpath in dirpaths:
    cmd1 = 'python main.py --mode test --data_root test\\'
    cmd2 = cmd1 +dirpath +' --batch_size 128 --max_dataset_size 9999 --gpu_ids -1 --ckpt_dir ckpts/ULMEGAN/210619_212934/ --load_epoch 40 --serial_batches --n_threads 0 --re_code --linear_fitting --pre_treatment --save_video'
    os.system(cmd2)
