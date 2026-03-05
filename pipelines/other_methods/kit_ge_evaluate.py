from glob import glob
import os
import shutil
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import IOUMatcher, CTCMatcher
from traccuracy.metrics import BasicMetrics, CTCMetrics
from traccuracy.utils import export_graphs_to_geff

gt_root = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION'
res_root = '/home/ddon0001/PhD/experiments/kit-ge'
CTC_METRICS = False

all_res = sorted(glob(os.path.join(res_root, '*', '*_RES')))

errors = []
for res_pth in all_res:
# for res_pth in ['/home/ddon0001/PhD/experiments/kit-ge/Fluo-N2DL-HeLa/02_RES']:
    gt_pth = res_pth.replace(res_root, gt_root).replace('_RES', '_GT/TRA')

    if CTC_METRICS:
        out_pth = res_pth.replace('_RES', '_ctc.zarr')
        matcher = CTCMatcher()
        metrics_class = CTCMetrics()
    else:
        out_pth = res_pth.replace('_RES', '_basic.zarr')
        matcher = IOUMatcher(iou_threshold=0.001, one_to_one=True)
        metrics_class = BasicMetrics()


    if os.path.exists(out_pth):
        shutil.rmtree(out_pth)
    try:
        res = load_ctc_data(res_pth)
        gt = load_ctc_data(gt_pth)
    except Exception as e:
        print(f"Error evaluating {res_pth}: {e}")
        errors.append(res_pth)
        continue

    matched = matcher.compute_mapping(gt, res)
    results = metrics_class.compute(matched)
    export_graphs_to_geff(out_pth, matched, [results])
for error in errors:
    print(f" - {error}")