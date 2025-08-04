import os

from generate_dataset_summary import generate_ctc_summary
from generate_configs import ds_summary_to_configs

ROOT_DIR = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/'
OUT_ROOT = '/home/ddon0001/PhD/experiments/misc/no_merges_small_k'
out_csv_path = os.path.join(OUT_ROOT, 'summary.csv')
# generate_ctc_summary(ROOT_DIR, out_csv_path)
configs = ds_summary_to_configs(
    out_csv_path,
    OUT_ROOT,
    div_constraint=False,
    allow_merges=False,
    k_neighbours=4
)

for config in configs:
    print(config.data_config.dataset_name)
    config.run(compute_additional_features=True)
    config.evaluate()
    print("#################################") 