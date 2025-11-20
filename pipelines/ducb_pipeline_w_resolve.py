# starting with "standard" merge-containing solution
# no prior edges have been inspected

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tracktour._tracker import VirtualVertices
from tracktour_experiments.generate_configs import get_config_for_row
from tracktour_experiments.ucb_policies import get_arm_to_play, initialize_bandit, get_count_arm_played, get_reward_for_arm

from tracktour_experiments.utils import update_segmentation_with_fn

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
        is_fp = "is_ctc_fp" in solution_graph.edges[u, v]
        is_ws = mark_ws_incorrect and ("is_wrong_semantic" in solution_graph.edges[u, v])
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
        if "is_ctc_fp" in solution_graph.edges[u, v]:
            return "FP"
        if "is_wrong_semantic" in solution_graph.edges[u, v]:
            return "WS"
        return "Correct"

    all_edges["oracle_is_correct"] = all_edges.apply(is_edge_tp, axis=1)
    all_edges["error_type"] = all_edges.apply(get_error_cat, axis=1)
    all_edges["solution_incorrect"] = all_edges.error_type != "Correct"


def get_edge_to_inspect(
        bandit_arms,
        sol_edges,
        feature_ranked,
        discounted_arm_played,
        discounted_arm_rewards,
        next_index,
        t,
        b,
        epsilon,
        gamma
    ):
    next_arm = get_arm_to_play(
        bandit_arms,
        discounted_arm_played,
        discounted_arm_rewards,
        B=b,
        epsilon=epsilon,
        gamma=gamma
    )
    edge = feature_ranked[next_arm].index[next_index[next_arm]]
    next_index[next_arm] += 1
    # this next edge has been seen by the bandit, with a different arm
    # reward the bandit, but select a new arm
    while sol_edges.loc[edge, 'bandit_rank'] != -1:
        discounted_arm_rewards[next_arm] += int(sol_edges.at[edge, 'solution_incorrect'])
        t += 1
        next_arm = get_arm_to_play(
            bandit_arms,
            discounted_arm_played,
            discounted_arm_rewards,
            B=b,
            epsilon=epsilon,
            gamma=gamma
        )
        edge = feature_ranked[next_arm].index[next_index[next_arm]]
        next_index[next_arm] += 1
    # now we have a new bandit edge to inspect
    sol_edges.loc[edge, 'bandit_rank'] = t
    sol_edges.loc[edge, 'bandit_arm'] = next_arm
    discounted_arm_rewards[next_arm] += int(sol_edges.at[edge, 'solution_incorrect'])
    t += 1
    return edge

def handle_edge(
        new_edge,
        sol_edges,
        tracker,
        gt_graph,
        pred_graph,
        sol_to_gt,
        gt_to_sol,
        sol_seg,
        gt_seg,
        edge_to_index,
        priority_queue,
        current_max_label:list[int],
        current_max_id:list[int],
        t_label_to_node_id
    ):
    edge_row = sol_edges.loc[new_edge]
    edge_index = int(edge_row.name)
    source = int(edge_row.u)
    target = int(edge_row.v)
    error_type = edge_row.error_type

    if error_type == 'Correct':
        # fix in model
        sol_edges.at[new_edge, 'fixed_edge'] = True
        tracker.fix_edge_in_model(edge_index, source, target, lb=1)
        return
    tracker.fix_edge_in_model(edge_index, source, target, lb=0, ub=0)
    # TODO: anything else?
    if error_type == 'FE':
        gt_successors = list(gt_graph.successors(sol_to_gt[target]))
        # add all FN successors (the first added successor gets put on the priority stack, and the rest get added "normally")
        fn_gt_successors = [succ for succ in gt_successors if succ not in gt_to_sol]
        added_fn = False
        for fn_succ in fn_gt_successors:
            # update segmentation with new FN label
            added_centroid = update_segmentation_with_fn(
                source,
                fn_succ,
                True,
                sol_to_gt,
                gt_graph,
                sol_seg,
                gt_seg,
                current_max_label,
                t_label_to_node_id
            )
            fn_label = current_max_label[0]
            fn_t = gt_graph.nodes[fn_succ]['t']
            current_max_id[0] += 1
            # at this point the segmentation is updated, but
            # nobody else knows about this vertex yet
            if not added_fn:
                # add vertex to model, connect edge from source, fix to 1
                # don't add migration/division/appearance edges yet
                # put it on the priority queue so we can check its 
                # successor stuff next
                # TODO: what are we actually putting on the queue?
                priority_queue.append(fn_succ)
            else:
                # to add to the tracker we need its label, t and centroid
                # we also pass in its new ID, in case removing nodes
                # messes things up
                tracker.add_vertex_to_model(
                    fn_label,
                    fn_t,
                    added_centroid,
                    current_max_id[0]
                )
                # add fn_succ to model as a brand new vertex
                # outgoing edges to 10 nearest neighbours
                # edge (D, fn_succ)
                # demand/capacity constraints
                added_fn = True
            # add fixed edge source, fn_succ to model

        # what gets added to the model for a new fn successor
            # flow(source, fn_succ)=1
            # if it's first added:
                # add to priority queue, nothing changed in model
            # else
                # add "normal" outgoing edges, including division incoming, but not appearance
        # mark existing predecessors of the true successors 0 
        if len(gt_successors) == 0:
            # this should never happen
            raise ValueError("No successor in GT for FE edge")
        if len(gt_successors) > 1:
            # TODO: handle division case
            # (we've already checked that there's no instances of 2 FN succs in CTC)
            pass
        else:
            # single GT successor
            gt_succ = gt_successors[0]
            if gt_succ in gt_to_sol:
                true_succ = gt_to_sol[gt_succ]
                # successor exists and it's the only one 
                # FE edge fix to 0, new edge LB=1, prior predecessor of true_succ fix to 0
                tracker.fix_edge_in_model(edge_index, source, target, lb=0, ub=0)
                if (source, true_succ) in edge_to_index:
                    tracker.fix_edge_in_model(edge_index, source, true_succ, lb=1)
                # working with no merges, so at most one predecessor (could be appearance)
                current_preds = sol_graph.predecessors()
                if len(current_preds) == 0:
                    current_pred = -2
                else:
                    current_pred = current_preds[0]
                tracker.fix_edge_in_model(edge_to_index[(current_pred, true_succ)], current_pred, true_succ, lb=0, ub=0)
            else:
                # single FN successor
                pass
        #  if correct successor exists, we just fix the edge into successor, mark current edge 0
        # whoever else was flowing into successor gets bumped, because we don't allow merges
        
        # if exactly one FN successor
            # introduce it, fix edge, fix exit to 0
            # we want to ensure next presented edge is (successor, exit) i.e. add faux FE edge again
        ...
        # if source is dividing into multiple FNs or two existing successors
            # introduce both FNs, fix edges to 1, fix (source, exit) to 0
            # ensure one of them is next, add other to model as a fresh vertex (or if it exists add FE edge)
        ...
    elif error_type == 'FP' or error_type == 'FA':
        pass
        # fix (source, target) to 0
        # find correct predecessor of target
        # (if it doesn't exist, introduce it with a faux appearance, next on the agenda)
        # fix the edge to 1


if __name__ == "__main__":
    ALLOW_MERGES = False
    MARK_WS_INCORRECT = False
    out_pth = "/home/ddon0001/PhD/experiments/scaled/pre-thesis/ducb_w_resolve_no_merges_no_ws"

    ds_summary_pth = (
        "/home/ddon0001/PhD/experiments/scaled/no_merges_all/summary.csv"
    )
    ds_summary = pd.read_csv(ds_summary_pth)

    ds_with_err = [
        # "Fluo-N3DH-SIM+_01",
        # "Fluo-N3DH-SIM+_02",
        # "Fluo-C3DL-MDA231_01",
        # "Fluo-C3DL-MDA231_02",
        # "Fluo-N2DH-GOWT1_01",
        # "Fluo-N2DH-GOWT1_02",
        # "PhC-C2DH-U373_01",
        # "PhC-C2DH-U373_02",
        "Fluo-N2DL-HeLa_01",
        # "Fluo-N2DL-HeLa_02",
        # "Fluo-C2DL-MSC_01",
        # "Fluo-C2DL-MSC_02",
        # "Fluo-C3DH-H157_02",
        # "DIC-C2DH-HeLa_01",
        # "DIC-C2DH-HeLa_02",
        # "Fluo-N3DH-CHO_01",
        # "Fluo-N3DH-CHO_02",
        # "BF-C2DL-MuSC_01",
        # "BF-C2DL-MuSC_02",
        # "BF-C2DL-HSC_01",
        # "BF-C2DL-HSC_02",
        # "Fluo-N2DH-SIM+_01",
        # "Fluo-N2DH-SIM+_02",
        # "PhC-C2DL-PSC_01",
        # "PhC-C2DL-PSC_02",
        # "Fluo-N3DH-CE_01",
        # "Fluo-N3DH-CE_02",
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
            allow_merges=ALLOW_MERGES,
        )
        # solve, keep tracked
        tracker, tracked = initial_config.run(compute_additional_features=True)
        # evaluate
        results, matched = initial_config.evaluate(
            tracked.tracked_detections, tracked.tracked_edges
        )
        gt_graph = matched.gt_graph.graph
        sol_graph = matched.pred_graph.graph
        gt_to_sol = {item[0]: item[1] for item in matched.mapping}
        sol_to_gt = {item[1]: item[0] for item in matched.mapping}
        t_label_to_node_id = {
            (node_info['t'], node_info['label']) : node_id 
            for node_id, node_info in sol_graph.nodes(data=True)
        }
        sol_seg = matched.pred_graph.segmentation
        gt_seg = matched.gt_graph.segmentation
        current_max_label = np.max(np.asarray(list(t_label_to_node_id.keys()))[:,1:]).reshape(1).tolist()
        current_max_id = [sol_graph.number_of_nodes() - 1]
        # assign edge error types
        populate_label_ws_enter_exit(
            tracked.all_edges,
            sol_graph,
            gt_graph,
            sol_to_gt,
            mark_ws_incorrect=MARK_WS_INCORRECT
        )
        tracked.all_edges.to_csv(
            f"{out_pth}/{ds_name}_all_edges_with_target_ws_fa_fe.csv", index=False
        )

        tracked.all_edges['fixed_edge'] = False
        tracked.all_edges['introduced_correction'] = False
        tracked.all_edges['bandit_rank'] = -1
        tracked.all_edges['bandit_arm'] = 'None'

        sol_edges = tracked.all_edges[
            (tracked.all_edges.flow > 0)
            & (tracked.all_edges.u != -1)
            & (tracked.all_edges.u != -3)
        ]
        edge_to_index = dict(zip(zip(tracked.all_edges['u'], tracked.all_edges['v']), tracked.all_edges.index))

        b = 2
        gamma = 1 - (1 / (4 * np.sqrt(2 * sol_edges.shape[0])))
        epsilon = 1/2

        bandit_arms=["cost", "softmax_entropy", "sensitivity_diff", "softmax", "parental_softmax"]
        ascending_sort=[False, False, True, True, True]      

        feature_ranked, played_ranks, rewards, next_index, t = initialize_bandit(sol_edges, bandit_arms, ascending_sort)
        discounted_arm_played = {
            arm: get_count_arm_played(played_ranks, arm, t, gamma) for arm in bandit_arms
        }
        discounted_arm_rewards = {
            arm: get_reward_for_arm(rewards, played_ranks, arm, t, gamma) for arm in bandit_arms
        }

        unsampled = set(sol_edges[sol_edges.bandit_rank == -1].index.tolist())
        priority_queue = []
        while len(unsampled) > 0:
            # print(f'Starting new inspection round for {ds_name}')
            # print(f'{len(unsampled)} edges still to inspect')
            if len(priority_queue):
                # TODO: what do we actually want to do with the priority queue?
                pass
            new_edge = get_edge_to_inspect(
                bandit_arms,
                sol_edges,
                feature_ranked,
                discounted_arm_played,
                discounted_arm_rewards,
                next_index,
                t,
                b,
                epsilon,
                gamma
            )
            handle_edge(
                new_edge,
                sol_edges,
                tracker,
                gt_graph,
                sol_graph,
                sol_to_gt,
                gt_to_sol,
                sol_seg,
                gt_seg,
                edge_to_index,
                priority_queue,
                current_max_label,
                current_max_id,
                t_label_to_node_id
            )
            # resolve every nth iteration or something
            


        #     if new_edge in unsampled:
        #         unsampled.remove(int(new_edge))
        # print("HI")



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
