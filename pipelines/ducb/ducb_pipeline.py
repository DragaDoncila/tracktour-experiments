import json
import os
import numpy as np
import pandas as pd
from geff import read_nx
from tqdm import tqdm
from tracktour._tracker import VirtualVertices
from tracktour_experiments.ucb_policies import rank_edges_by_ucb

def populate_label_ws_enter_exit(
        all_edges,
        solution_graph,
        gt_graph,
        sol_to_gt,
        mark_ws_incorrect=True
    ):
    def is_edge_tp(edge):
        u, v = int(edge["u"]), int(edge["v"])
        # appearance edge, correct if destination node in gt_graph
        # has incoming degree 0
        if u == VirtualVertices.APP.value:
            return int(gt_graph.in_degree(sol_to_gt[v]) == 0)
        # exit edge, correct if source node in gt_graph
        # has outgoing degree 0
        elif v == VirtualVertices.TARGET.value:
            return int(gt_graph.out_degree(sol_to_gt[u]) == 0)
        elif not solution_graph.has_edge(u, v):
            return 0
        is_fp = solution_graph.edges[u, v].get("EdgeFlag.CTC_FALSE_POS", False)
        is_ws = mark_ws_incorrect and solution_graph.edges[u, v].get("EdgeFlag.WRONG_SEMANTIC", False)
        is_tp = not (is_fp or is_ws)
        return int(is_tp)

    def get_error_cat(edge):
        u, v = int(edge["u"]), int(edge["v"])
        if u == VirtualVertices.APP.value:
            if edge["flow"] <= 0:
                return "None"
            if not edge["oracle_is_correct"]:
                return "FA"
            return "Correct"
        if v == VirtualVertices.TARGET.value:
            if edge["flow"] <= 0:
                return "None"
            if not edge["oracle_is_correct"]:
                return "FE"
            return "Correct"
        if not solution_graph.has_edge(u, v):
            return "None"
        if solution_graph.edges[u, v].get("EdgeFlag.CTC_FALSE_POS", False):
            return "FP"
        if solution_graph.edges[u, v].get("EdgeFlag.WRONG_SEMANTIC", False):
            return "WS"
        return "Correct"

    all_edges["oracle_is_correct"] = all_edges.apply(is_edge_tp, axis=1)
    all_edges["error_type"] = all_edges.apply(get_error_cat, axis=1)
    all_edges["solution_incorrect"] = all_edges.oracle_is_correct == 0

def read_traccuracy_graphs(ds_dir):
    map_path = os.path.join(ds_dir, 'matching.json')
    zarr_path = os.path.join(ds_dir, 'matched_solution.zarr')

    sol_path = os.path.join(zarr_path, 'pred.geff')
    gt_path = os.path.join(zarr_path, 'gt.geff')
    sol_graph = read_nx(sol_path, validate=False)[0]
    gt_graph = read_nx(gt_path, validate=False)[0]
    with open(map_path, 'r') as map_file:
        matching = json.load(map_file)
    return sol_graph, gt_graph, matching

def populate_errors_all_datasets(solved_ds_dir, out_dir, mark_ws_incorrect=True):
    ds_names = [
        name for name in os.listdir(solved_ds_dir)
        if os.path.isdir(os.path.join(solved_ds_dir, name))
    ]
    for ds_name in tqdm(ds_names):
        ds_dir = os.path.join(solved_ds_dir, ds_name)
        all_edges_pth = os.path.join(ds_dir, 'all_edges.csv')
        all_edges = pd.read_csv(all_edges_pth)
        sol_graph, gt_graph, mapping = read_traccuracy_graphs(ds_dir)
        sol_to_gt = {tup[1] : tup[0] for tup in mapping}

        populate_label_ws_enter_exit(
            all_edges,
            sol_graph,
            gt_graph,
            sol_to_gt,
            mark_ws_incorrect=mark_ws_incorrect
        )
        all_edges.to_csv(f'{out_dir}/{ds_name}_all_edges_with_errors.csv', index=False)

if __name__ == "__main__":
    MARK_WS_INCORRECT = False
    SOLVED_DS_DIR = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/scaled_w_merge'
    OUT_DF_PTH = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/ducb/merges_no_ws_cost_softmax_only'

    # assign error categories to edges in solved datasets
    populate_errors_all_datasets(SOLVED_DS_DIR, OUT_DF_PTH, mark_ws_incorrect=MARK_WS_INCORRECT)

    all_df_pths = [
        os.path.join(OUT_DF_PTH, f) for f in os.listdir(OUT_DF_PTH) if f.endswith('.csv')
    ]

    for pth in tqdm(all_df_pths):
        ds_name = os.path.basename(pth).replace('_all_edges_with_errors.csv', '')
        ds_df = pd.read_csv(pth)
        ds_df['bandit_rank'] = -1
        ds_df['bandit_arm'] = 'None'
        # in solution, not source, not div
        sol_df = ds_df[(ds_df.flow > 0) & (ds_df.u != -1) & (ds_df.u != -3)]

        b = 2
        gamma = 1 - (1 / (4 * np.sqrt(2 * ds_df.shape[0])))
        epsilon = 1/2
        print('Processing', ds_name, 'with gamma', gamma)
        rank_edges_by_ucb(
            sol_df,
            bandit_arms=[
                "cost",
                # "softmax_entropy",
                # "sensitivity_diff",
                "softmax"
            ],
            ascending_sort=[
                False,
                # False,
                # True,
                True
            ],
            B=b,
            epsilon=epsilon,
            gamma=gamma
        )
        ds_df.loc[sol_df.index, 'bandit_rank'] = sol_df.bandit_rank
        ds_df.loc[sol_df.index, 'bandit_arm'] = sol_df.bandit_arm
        ds_df.to_csv(pth, index=False)
        print(f'Wrote {pth}')
        print('#' * 40)