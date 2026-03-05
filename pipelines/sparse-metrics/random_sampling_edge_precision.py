"""This module explores how random edge sampling precision estimates
converge on the true dataset precision.

We generate 30 random sampling orders of the edges in the solution.
We iterate through the random edges in each order, and compute
the cumulative precision estimate, and standard error of the estimate.

For each dataset, iteration and edge we save:
- the dataset name
- the sample ID
- the edge ID
- whether that edge is TP or FP
- the cumulative precision estimate at that edge
- the standard error of the estimate at that edge

to a single CSV file for later analysis.
"""

import json
import os
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from traccuracy.loaders import load_geff_data

def sample_edges_random(sol):
    edge_ids = np.asarray(sol.edges)
    sample_order = np.random.permutation(edge_ids)
    return sample_order

def get_estimates(population_size, count_sampled, count_sampled_tp):
    if count_sampled == 0:
        return 0, 0
    prec_estimate, std_error = get_estimates(population_size, count_sampled, count_sampled_tp)
    finite_population_correction = np.sqrt((population_size - count_sampled) / (population_size  - 1))
    prec_estimate = count_sampled_tp / count_sampled
    sample_sd = np.sqrt(prec_estimate * (1 - prec_estimate))
    std_error = finite_population_correction * (sample_sd / np.sqrt(count_sampled))
    return prec_estimate, std_error

if __name__ == '__main__':
    out_root = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/sparse_sampling/'
    sol_root = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/scaled_no_merge_cap/'
    all_ds_names = [item for item in os.listdir(sol_root) if os.path.isdir(os.path.join(sol_root, item))]
    
    # for ds_name in tqdm(all_ds_names):
    for ds_name in tqdm(['Fluo-N2DL-HeLa_01']):
        sol_path = sol_root + ds_name + '/'

        sol = load_geff_data(
            f'{sol_path}/matched_solution.zarr/pred.geff',
            load_all_props=True
        ).graph

        population_size = sol.number_of_edges()

        sample_order = sample_edges_random(sol)
        count_sampled = 0
        count_sampled_tp = 0
        count_sampled_fp = 0

        for i, edge in enumerate(sample_order):
            # no FP nodes so this can't fail
            edge_info = sol.edges[edge]
            if edge_info.get('tp', 0):
                count_sampled_tp += 1
            elif edge_info.get('fp', 0):
                count_sampled_fp += 1
            else:
                raise ValueError(f'Edge {edge} in {ds_name} is neither TP nor FP')
            prec_estimate, std_error = get_estimates(population_size, count_sampled, count_sampled_tp)
