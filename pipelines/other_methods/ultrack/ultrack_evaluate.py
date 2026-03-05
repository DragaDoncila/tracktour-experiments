from glob import glob
import os
import shutil
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import IOUMatcher, CTCMatcher
from traccuracy.metrics import BasicMetrics, CTCMetrics
from traccuracy.utils import export_graphs_to_geff

gt_root = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION'
res_root = '/home/ddon0001/PhD/experiments/ultrack'
CTC_METRICS = True

suffix = '_default'
all_res = sorted(glob(os.path.join(res_root, f'*{suffix}_RES')))

errors = {}
for res_pth in all_res:
# for res_pth in ['/home/ddon0001/PhD/experiments/kit-ge/Fluo-N2DL-HeLa/02_RES']:
    res_ds_name = os.path.basename(res_pth).removesuffix(f'{suffix}_RES')
    res_ds, res_seq = res_ds_name.split('_')
    gt_pth = f'{gt_root}/{res_ds}/{res_seq}_GT/TRA/'
    
    if CTC_METRICS:
        out_pth = res_pth.replace(f'{suffix}_RES', f'{suffix}_ctc.zarr')
        matcher = CTCMatcher()
        metrics_class = CTCMetrics()
    else:
        out_pth = res_pth.replace(f'{suffix}_RES', f'{suffix}_basic.zarr')
        matcher = IOUMatcher(iou_threshold=0.001, one_to_one=True)
        metrics_class = BasicMetrics()


    if os.path.exists(out_pth):
        continue
    try:
        res = load_ctc_data(res_pth)
        gt = load_ctc_data(gt_pth)
    except Exception as e:
        print(f"Error evaluating {res_pth}: {e}")
        errors[res_pth] = str(e)
        continue

    matched = matcher.compute_mapping(gt, res)
    results = metrics_class.compute(matched)
    export_graphs_to_geff(out_pth, matched, [results])
for error in errors:
    print(f"{error} - {errors[error]}")