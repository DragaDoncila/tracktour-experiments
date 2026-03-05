import os
import subprocess

ds_dir = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION'
res_root_dir = '/home/ddon0001/PhD/experiments/kit-ge/'

datasets = [ds for ds in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, ds)) and ds != 'SW']

err_seg_paths = [(ds, seq) for ds in datasets for seq in os.listdir(os.path.join(ds_dir, ds)) if 'ERR_SEG' in seq]

for ds, seq in err_seg_paths:
    seq_num = seq.split('_')[0]
    res_dir = os.path.join(res_root_dir, ds, f'{seq_num}_RES')
    im_pth = os.path.join(ds_dir, ds, seq_num)
    seg_pth = os.path.join(ds_dir, ds, seq)
    if os.path.exists(res_dir):
        print(f'Skipping existing results: {res_dir}')
        continue
    os.makedirs(res_dir, exist_ok=True)
    print('-'*50)
    print(f'Processing dataset: {ds}, sequence: {seq}')
    cmd = [
        'python',
        # '-m',
        '/home/ddon0001/PhD/code/KIT-GE-3-Cell-Tracking-for-CTC/run_tracking.py',
        '--image_path',
        im_pth,
        '--segmentation_path',
        seg_pth,
        '--results',
        res_dir,
    ]
    subprocess.run(cmd, check=True)