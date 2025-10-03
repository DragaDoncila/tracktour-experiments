# starting with "standard" merge-containing solution
# no prior edges have been inspected

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tracktour._tracker import VirtualVertices
from tracktour_experiments.generate_configs import get_config_for_row
from tracktour_experiments.ucb_policies import get_arm_to_play, initialize_bandit


def populate_label_ws_enter_exit(all_edges, solution_graph, gt_graph, sol_to_gt):
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
        is_tp = ("is_ctc_fp" not in solution_graph.edges[u, v]) and (
            "is_wrong_semantic" not in solution_graph.edges[u, v]
        )
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
        if "is_ctc_fp" in solution_graph.edges[u, v]:
            return "FP"
        if "is_wrong_semantic" in solution_graph.edges[u, v]:
            return "WS"
        return "Correct"

    all_edges["oracle_is_correct"] = all_edges.apply(is_edge_tp, axis=1)
    all_edges["error_type"] = all_edges.apply(get_error_cat, axis=1)
    all_edges["solution_incorrect"] = all_edges.error_type != "Correct"


def get_edge_to_inspect(
        still_to_inspect,
        sol_edges,
        feature_ranked,
        played_ranks,
        rewards,
        next_index,
        t,
        b,
        epsilon,
        gamma
    ):
    next_arm = get_arm_to_play(
        still_to_inspect,
        played_ranks,
        rewards,
        t,
        B=b,
        epsilon=epsilon,
        gamma=gamma
    )
    edge = feature_ranked[next_arm].index[next_index[next_arm]]
    next_index[next_arm] += 1
    # this next edge has been seen by the bandit, with a different arm
    # reward the bandit, but select a new arm
    while sol_edges.loc[edge, 'bandit_rank'] != -1:
        played_ranks[next_arm].append(t)
        rewards[next_arm].append(int(sol_edges.loc[edge, 'solution_incorrect']))
        t += 1
        next_arm = get_arm_to_play(
            still_to_inspect,
            played_ranks,
            rewards,
            t,
            B=b,
            epsilon=epsilon,
            gamma=gamma
        )
        edge = feature_ranked[next_arm].index[next_index[next_arm]]
        next_index[next_arm] += 1
    # now we have a new bandit edge to inspect
    sol_edges.loc[edge, 'bandit_rank'] = t
    sol_edges.loc[edge, 'bandit_arm'] = next_arm
    played_ranks[next_arm].append(t)
    rewards[next_arm].append(int(sol_edges.loc[edge, 'solution_incorrect']))
    t += 1
    return edge

def correct_edge(new_edge, sol_edges, tracked, gt_graph, pred_graph, sol_to_gt, gt_to_sol):
    edge_row = sol_edges.loc[new_edge]
    source = edge_row.u
    error_type = edge_row.error_type

    gt_vertices_to_add = []
    edges_to_remove = []
    edges_to_add = []

    if error_type == 'FA':
        pass
    elif error_type == 'FE':
        pass
    else:
        pass
        



if __name__ == "__main__":
    out_pth = "/home/ddon0001/PhD/experiments/scaled/pre-thesis/ducb_w_resolve"

    ds_summary_pth = (
        "/home/ddon0001/PhD/experiments/scaled/no_div_constraint_err_seg/summary.csv"
    )
    ds_summary = pd.read_csv(ds_summary_pth)

    ds_with_err = [
        "Fluo-N3DH-SIM+_01",
        "Fluo-N3DH-SIM+_02",
        "Fluo-C3DL-MDA231_01",
        "Fluo-C3DL-MDA231_02",
        "Fluo-N2DH-GOWT1_01",
        "Fluo-N2DH-GOWT1_02",
        "PhC-C2DH-U373_01",
        "PhC-C2DH-U373_02",
        "Fluo-N2DL-HeLa_01",
        "Fluo-N2DL-HeLa_02",
        "Fluo-C2DL-MSC_01",
        "Fluo-C2DL-MSC_02",
        "Fluo-C3DH-H157_02",
        "DIC-C2DH-HeLa_01",
        "DIC-C2DH-HeLa_02",
        "Fluo-N3DH-CHO_01",
        "Fluo-N3DH-CHO_02",
        "BF-C2DL-MuSC_01",
        "BF-C2DL-MuSC_02",
        "BF-C2DL-HSC_01",
        "BF-C2DL-HSC_02",
        "Fluo-N2DH-SIM+_01",
        "Fluo-N2DH-SIM+_02",
        "PhC-C2DL-PSC_01",
        "PhC-C2DL-PSC_02",
        "Fluo-N3DH-CE_01",
        "Fluo-N3DH-CE_02",
    ]

    feature_names = [
        "cost",
        "softmax_entropy",
        "sensitivity_diff",
        "softmax",
        "parental_softmax",
    ]
    for ds_name in tqdm(ds_with_err):

        ds_summary_row = ds_summary[ds_summary["ds_name"] == ds_name].squeeze()
        initial_config = get_config_for_row(
            ds_summary_row,
            out_root=out_pth,
            div_constraint=False,
        )
        # solve, keep tracked
        tracked = initial_config.run(compute_additional_features=True)
        # evaluate
        results, matched = initial_config.evaluate(
            tracked.tracked_detections, tracked.tracked_edges
        )
        gt_graph = matched.gt_graph.graph
        sol_graph = matched.pred_graph.graph
        gt_to_sol = {item[0]: item[1] for item in matched.mapping}
        sol_to_gt = {item[1]: item[0] for item in matched.mapping}
        # assign edge error types
        populate_label_ws_enter_exit(
            tracked.all_edges,
            sol_graph,
            gt_graph,
            sol_to_gt,
        )
        tracked.all_edges.to_csv(
            f"{out_pth}/{ds_name}_all_edges_with_target_ws_fa_fe.csv", index=False
        )

        tracked.all_edges['inspected'] = False
        tracked.all_edges['fixed_edge'] = False
        tracked.all_edges['introduced_correction'] = False
        tracked.all_edges['bandit_rank'] = -1
        tracked.all_edges['bandit_arm'] = 'None'

        sol_edges = tracked.all_edges[
            (tracked.all_edges.flow > 0)
            & (tracked.all_edges.u != -1)
            & (tracked.all_edges.u != -3)
        ]

        b = 2
        gamma = 1 - (1 / (4 * np.sqrt(2 * sol_edges.shape[0])))
        epsilon = 1/2

        bandit_arms=["cost", "softmax_entropy", "sensitivity_diff", "softmax", "parental_softmax"]
        ascending_sort=[False, False, True, True, True]      

        feature_ranked, played_ranks, rewards, next_index, t = initialize_bandit(
            sol_edges,
            bandit_arms=bandit_arms,
            ascending_sort=ascending_sort
        )
        while sol_edges.inspected.sum() < sol_edges.shape[0]:
            print(f'Starting new inspection round for {ds_name}')
            still_to_inspect = sol_edges[~sol_edges.inspected]
            print(f'{still_to_inspect.shape[0]} edges still to inspect')
            new_edge = get_edge_to_inspect(
                still_to_inspect,
                sol_edges,
                feature_ranked,
                played_ranks,
                rewards,
                next_index,
                t,
                b,
                epsilon,
                gamma
            )
            # if the edge is incorrect, we correct it
            if sol_edges.loc[new_edge, 'solution_incorrect']:
                correct_edge(new_edge, sol_edges, tracked, gt_graph, sol_graph, sol_to_gt, gt_to_sol)
            # otherwise we fix it in the model and move on
            else:
                fix_edge_in_model(new_edge, sol_edges, tracked)





        # get ranked edges for each feature
        # assign DUCB rank to each solution edge
        # solution_edges = tracked.all_edges[tracked.all_edges.flow > 0]
        # edges_with_ducb_rank = populate_ducb_ranking(solution_edges, feature_names)
        
        
        # sample n edges and correct errors
        # evaluate & save
        # fix inspected edges
        # re-solve & save
        # evaluate & save
        # update the config...?
        # repeat until we've inspected all edges?
