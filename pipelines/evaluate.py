import os
from generate_configs import ds_summary_to_configs


ROOT_DIR = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/'
OUT_ROOT = '/home/ddon0001/PhD/experiments/flow_penalty/st_50/'
out_csv_path = os.path.join(OUT_ROOT, 'summary.csv')


configs = ds_summary_to_configs(out_csv_path, OUT_ROOT, div_constraint=False, penalize_flow=True)

for config in configs:
    if not os.path.exists(os.path.join(config.data_config.out_root_path, config.data_config.dataset_name, 'metrics.json')):
        config.evaluate()
    else:
        print('Finished', config.data_config.dataset_name)