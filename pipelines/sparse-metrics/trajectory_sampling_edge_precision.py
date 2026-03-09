"""This module explores how trajectory-based edge sampling precision estimates
converge on the true dataset precision.

We simulate a biologist sampling connected components (lineages) using DFS:
starting from randomly permuted root nodes, following one branch fully before
backtracking to the other at each division.

We generate 30 DFS sampling orders. For each, we iterate through edges in that
order, computing cumulative precision estimate and standard error as we go.

For each dataset, run and edge we save:
- the dataset name
- the run_id
- the sample_id (position in the DFS order)
- the edge src and tgt
- whether that edge is TP or FP
- the cumulative precision estimate at that edge
- the standard error of the estimate at that edge

to one CSV file per dataset.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from traccuracy.loaders import load_geff_data


def sample_edges_dfs(sol):
    """Returns edges in DFS order starting from randomly permuted root nodes.
    At each division, randomly chooses which daughter branch to follow first."""
    roots = [n for n in sol.nodes if sol.in_degree(n) == 0]
    np.random.shuffle(roots)

    edge_order = []
    visited_nodes = set()
    visited_edges = set()

    for root in roots:
        if root in visited_nodes:
            continue
        visited_nodes.add(root)
        succs = list(sol.successors(root))
        np.random.shuffle(succs)
        # push onto stack in reverse order so we can use .pop()
        stack = [(root, s) for s in reversed(succs)]

        while stack:
            parent, child = stack.pop()
            edge = (parent, child)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            edge_order.append(edge)

            if child not in visited_nodes:
                visited_nodes.add(child)
                child_succs = list(sol.successors(child))
                np.random.shuffle(child_succs)
                for s in reversed(child_succs):
                    stack.append((child, s))

    return edge_order


def get_estimates(population_size, count_sampled, count_sampled_tp):
    if count_sampled == 0:
        return 0, 0
    finite_population_correction = np.sqrt((population_size - count_sampled) / (population_size - 1))
    prec_estimate = count_sampled_tp / count_sampled
    sample_sd = np.sqrt(prec_estimate * (1 - prec_estimate))
    std_error = finite_population_correction * (sample_sd / np.sqrt(count_sampled))
    return prec_estimate, std_error


if __name__ == '__main__':
    out_root = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/sparse_sampling/edge_precision_trajectory'
    sol_root = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/scaled_no_merge_cap/'
    all_ds_names = [item for item in os.listdir(sol_root) if os.path.isdir(os.path.join(sol_root, item))]

    for ds_name in tqdm(all_ds_names):
    # for ds_name in tqdm(['Fluo-N2DL-HeLa_01']):
        sol_path = sol_root + ds_name + '/'

        sol = load_geff_data(
            f'{sol_path}/matched_solution.zarr/pred.geff',
            load_all_props=True
        ).graph

        population_size = sol.number_of_edges()
        records = []

        for run_id in range(30):
            sample_order = sample_edges_dfs(sol)
            count_sampled = 0
            count_sampled_tp = 0
            count_sampled_fp = 0

            for i, edge in enumerate(sample_order):
                edge_info = sol.edges[edge]
                is_tp = bool(edge_info.get('tp', 0))
                if is_tp:
                    count_sampled_tp += 1
                elif edge_info.get('fp', 0):
                    count_sampled_fp += 1
                else:
                    raise ValueError(f'Edge {edge} in {ds_name} is neither TP nor FP')
                count_sampled += 1
                prec_estimate, std_error = get_estimates(population_size, count_sampled, count_sampled_tp)
                records.append({
                    'ds_name': ds_name,
                    'run_id': run_id,
                    'sample_id': i,
                    'edge_src': edge[0],
                    'edge_tgt': edge[1],
                    'is_tp': is_tp,
                    'prec_estimate': prec_estimate,
                    'std_error': std_error,
                })

        pd.DataFrame(records).to_csv(f'{out_root}/{ds_name}.csv', index=False)
