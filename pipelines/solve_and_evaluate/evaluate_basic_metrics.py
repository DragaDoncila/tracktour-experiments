import os
import shutil
import pandas as pd
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import IOUMatcher
from traccuracy.metrics import BasicMetrics
from traccuracy.utils import export_graphs_to_geff


gt_dir = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION'
res_dir = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/ctc_format/scaled_ctc'

ds_summary = pd.read_csv(f'{res_dir}/summary.csv')

for row in ds_summary.itertuples():
    print(f'Processing dataset: {row.ds_name}')
    ds_name = row.ds_name
    ds, seq = ds_name.split('_')
    res_pth = f'{res_dir}/{ds_name}/RES/'
    gt_pth = f'{gt_dir}/{ds}/{seq}_GT/TRA/'
    out_zarr = f'{res_dir}/{ds_name}/basic_metrics.zarr'

    if os.path.exists(out_zarr):
        shutil.rmtree(out_zarr)

    gt = load_ctc_data(gt_pth)
    res = load_ctc_data(res_pth)

    matcher = IOUMatcher(iou_threshold=0.001, one_to_one=True)
    matched = matcher.compute_mapping(gt, res)
    results = BasicMetrics().compute(matched)
    
    export_graphs_to_geff(out_zarr, matched, [results])
