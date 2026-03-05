
import json
import networkx as nx
import numpy as np
import os
import pandas as pd

from geff import read_nx
from tifffile import imwrite
from tqdm import tqdm
from tracktour import load_tiff_frames
from tracktour_experiments.generate_dataset_summary import generate_ctc_summary
from tracktour_experiments.generate_configs import ds_summary_to_configs

def get_merge_info_for_sol(sol_graph, gt_graph, matching):
    merge_nodes = []
    merge_lengths = []
    merge_fates = []
    merge_fate_correct = []
    merge_exit_node = []

    pred_to_gt = {tup[1] : tup[0] for tup in matching}
    
    merges = [node for node in sol_graph.nodes() if sol_graph.in_degree(node) > 1]
    # check how long the merge continues for
    for merge in merges:
        merge_length = 0
        current_node = merge
        # while the node only has at most one successor, we continue
        while sol_graph.out_degree(current_node) == 1:
            merge_length += 1
            # move to the next node in the merge
            current_node = list(sol_graph.successors(current_node))[0]
        # current_node is the end of the merge. Check if it's terminating or dividing
        if sol_graph.out_degree(current_node) == 0:
            merge_fate = 'terminate'
            fate_correct = False
            if current_node in pred_to_gt:
                gt_node = pred_to_gt[current_node]
                if gt_graph.out_degree(gt_node) == 0:
                    fate_correct = True
                else:
                    print(f"{gt_node} out degree {gt_graph.out_degree(gt_node)}")
            else:
                raise ValueError(f"Node {current_node} not found in predicted to ground truth mapping.")
        elif sol_graph.out_degree(current_node) > 1:
            merge_fate = 'divide'
            fate_correct = True
            for edge in sol_graph.out_edges(current_node):
                e_info = sol_graph.edges[edge]
                old_fp = e_info.get('EdgeFlag.FALSE_POS', False)
                old_ws = e_info.get('EdgeFlag.WRONG_SEMANTIC', False)
                new_fp = e_info.get('EdgeFlag.CTC_FALSE_POS', False)
                if old_fp or old_ws or new_fp:
                    fate_correct = False
        else:
            raise ValueError(f"Unexpected out-degree for merge end node {current_node}: {sol_graph.out_degree(current_node)}")
        # store the merge information
        merge_nodes.append(merge)
        merge_lengths.append(merge_length)
        merge_fates.append(merge_fate)
        merge_fate_correct.append(fate_correct)
        merge_exit_node.append(current_node)
    return merge_nodes, merge_lengths, merge_fates, merge_fate_correct, merge_exit_node

def flows_into_fn(merge_node, sol_graph, gt_graph, pred_to_gt):
    """
    Check if the merge node flows into an FN node in the ground truth graph.
    """
    merge_parents = list(sol_graph.predecessors(merge_node))
    for parent in merge_parents:
        if parent in pred_to_gt:
            gt_node = pred_to_gt[parent]
            gt_successors = list(gt_graph.successors(gt_node))
            # if there's multiple successors for GT parent, this is a little wonky
            # but presumably would still be identified by user?
            for successor in gt_successors:
                succ_info = gt_graph.nodes[successor]
                is_fn = succ_info.get('NodeFlag.FALSE_NEG', False) or succ_info.get('NodeFlag.CTC_FALSE_NEG', False)
                if is_fn:
                    return is_fn, successor
    # either no parent has FN successors, or no parent has successors
    return False, -1

def save_tiff_frames(out_dir, masks):
    os.makedirs(out_dir, exist_ok=True)
    n_digits = max(len(str(len(masks))), 3)
    for i, frame in enumerate(masks):
        frame_out_name = os.path.join(out_dir, f"mask{str(i).zfill(n_digits)}.tif")
        imwrite(frame_out_name, frame, compression="zlib")

def load_sol_info(name, sol_root, seg_path, gt_seg_path, as_zarr=False):
    map_path = os.path.join(sol_root, name, 'matching.json')
    if as_zarr:
        zarr_path = os.path.join(sol_root, name, 'matched_solution.zarr')
        sol_path = os.path.join(zarr_path, 'pred.geff')
        gt_path = os.path.join(zarr_path, 'gt.geff')
        sol_graph = read_nx(sol_path, validate=False)[0]
        gt_graph = read_nx(gt_path, validate=False)[0]        
    else:
        sol_path = os.path.join(sol_root, name, 'matched_solution.graphml')
        gt_path = os.path.join(sol_root, name, 'matched_gt.graphml')
        sol_graph = nx.read_graphml(sol_path, node_type=int)
        gt_graph = nx.read_graphml(gt_path)

    seg = load_tiff_frames(seg_path)
    gt = load_tiff_frames(gt_seg_path)

    with open(map_path, 'r') as map_file:
        matching = json.load(map_file)
    pred_to_gt = {tup[1] : tup[0] for tup in matching}
    return sol_graph, gt_graph, seg, gt, pred_to_gt

# step 1: find all merges in the solution, see how many have parents flowing into fn vertex
# and check overlaps of the GT fn vertex, if applicable
# skipping all skip-edge connections
# save that info out to merge file
def get_basic_merge_df_for_sol_set(sol_root, as_zarr=False):
    all_ds = [dir_name for dir_name in os.listdir(sol_root) if os.path.isdir(os.path.join(sol_root, dir_name))]
    
    ds_names = []
    merge_nodes = []
    merge_lengths = []
    merge_fates = []
    merge_fate_correct = []
    merge_exit_node = []

    for ds in tqdm(all_ds):
        if as_zarr:
            sol_path = os.path.join(sol_root, ds, 'matched_solution.zarr', 'pred.geff')
            gt_path = os.path.join(sol_root, ds, 'matched_solution.zarr', 'gt.geff')

            sol_graph, _ = read_nx(sol_path, validate=False)
            gt_graph, _ = read_nx(gt_path, validate=False)
        else:
            sol_path = os.path.join(sol_root, ds, 'matched_solution.graphml')
            gt_path = os.path.join(sol_root, ds, 'matched_gt.graphml')
            sol_graph = nx.read_graphml(sol_path, node_type=int)
            gt_graph = nx.read_graphml(gt_path)

        map_path = os.path.join(sol_root, ds, 'matching.json')
        with open(map_path, 'r') as map_file:
            matching = json.load(map_file)

        current_merge_nodes, current_merge_lengths, current_merge_fates, current_merge_fate_correct, current_merge_exit_node = get_merge_info_for_sol(sol_graph, gt_graph, matching)

        ds_names.extend([ds for _ in range(len(current_merge_nodes))])
        merge_nodes.extend(current_merge_nodes)
        merge_lengths.extend(current_merge_lengths)
        merge_fates.extend(current_merge_fates)
        merge_fate_correct.extend(current_merge_fate_correct)
        merge_exit_node.extend(current_merge_exit_node)
    
    merge_df = pd.DataFrame({
        'ds_name': ds_names,
        'merge_node': merge_nodes,
        'merge_length': merge_lengths,
        'merge_fate': merge_fates,
        'merge_fate_correct': merge_fate_correct,
        'merge_exit_node': merge_exit_node
    })
    return merge_df

def populate_merge_resolution_info_for_sol_set(merge_df, sol_root, as_zarr=False):
    """Populate merge resolution information for a set of solutions.

    We check if the matched ground truth successors of the merge nodes are 
    FN nodes. 

    Args:
        merge_df (_type_): _description_
        sol_root (_type_): _description_
        as_zarr (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    merge_df['parent_has_fn_succ'] = False
    merge_df['fn_succ'] = '-1'
    for name, group in tqdm(merge_df.groupby('ds_name')):
        if as_zarr:
            sol_path = os.path.join(sol_root, name, 'matched_solution.zarr', 'pred.geff')
            gt_path = os.path.join(sol_root, name, 'matched_solution.zarr', 'gt.geff')

            sol_graph, _ = read_nx(sol_path, validate=False)
            gt_graph, _ = read_nx(gt_path, validate=False)
        else:
            sol_path = os.path.join(sol_root, name, 'matched_solution.graphml')
            gt_path = os.path.join(sol_root, name, 'matched_gt.graphml')
            sol_graph = nx.read_graphml(sol_path, node_type=int)
            gt_graph = nx.read_graphml(gt_path)

        map_path = os.path.join(sol_root, name, 'matching.json')
        with open(map_path, 'r') as map_file:
            matching = json.load(map_file)
        pred_to_gt = {tup[1] : tup[0] for tup in matching}
        for row in group.itertuples():
            is_fn, fn_succ = flows_into_fn(row.merge_node, sol_graph, gt_graph, pred_to_gt)
            merge_df.loc[row.Index, 'parent_has_fn_succ'] = bool(is_fn)
            merge_df.loc[row.Index, 'fn_succ'] = fn_succ
    return merge_df

def populate_overlap_info_for_sol_set(merge_df, summary_df, sol_root, as_zarr=False):
    merge_df['fn_overlaps'] = False
    can_introduce = merge_df[merge_df.parent_has_fn_succ == True]
    for name, group in tqdm(can_introduce.groupby('ds_name')):
        # we need segmentation and GT segmentation
        summary_info = summary_df[summary_df.ds_name == name]
        seg_path = summary_info.seg_path.values[0]
        gt_seg_path = summary_info.tra_gt_path.values[0]
        seg = load_tiff_frames(seg_path)
        gt = load_tiff_frames(gt_seg_path)

        if as_zarr:
            sol_path = os.path.join(sol_root, name, 'matched_solution.zarr', 'pred.geff')
            gt_path = os.path.join(sol_root, name, 'matched_solution.zarr', 'gt.geff')

            sol_graph, _ = read_nx(sol_path, validate=False)
            gt_graph, _ = read_nx(gt_path, validate=False)
        else:
            sol_path = os.path.join(sol_root, name, 'matched_solution.graphml')
            gt_path = os.path.join(sol_root, name, 'matched_gt.graphml')
            sol_graph = nx.read_graphml(sol_path, node_type=int)
            gt_graph = nx.read_graphml(gt_path)


        for row in group.itertuples():
            fn_succ = row.fn_succ
            fn_info = gt_graph.nodes[fn_succ]
            fn_t = fn_info['t']
            fn_label = fn_info['segmentation_id']

            merge_v = row.merge_node
            merge_info = sol_graph.nodes[merge_v]
            merge_t = merge_info['t']

            if fn_t != merge_t:
                print(f"FN succ of merge parent for {name} is not in merge frame. Likely skip edge!")
                continue

            seg_binary_frame = seg[fn_t] > 0
            gt_label_mask = gt[fn_t] == fn_label
            # do we have any overlaps?
            has_overlap = np.logical_and(seg_binary_frame, gt_label_mask).any()
            merge_df.loc[row.Index, 'fn_overlaps'] = has_overlap
    return merge_df

def replace_fn_and_overlapping_mask(
        fn_frame,
        seg,
        merge_t,
        fn_label,
        t_label_to_node_id,
        new_id,
        pred_to_gt,
        gt_graph
    ):
    # if fn overlaps any existing seg ID, find the GT version of that seg ID
    # delete the original seg ID, and copy over the fn label mask and the GT version
    # of overlapping ID
    merge_frame = seg[merge_t]
    fn_mask = fn_frame == fn_label
    overlapping_mask = (merge_frame > 0) & (fn_mask > 0)
    overlapping_indices = overlapping_mask.nonzero()
    # what's the seg value that we're overlapping with in seg
    overlapping_seg_ids = merge_frame[overlapping_indices]
    # we might overlap with multiple existing SEG ids e.g. the merge, and another one
    # we replace each overlapping ID with their GT matched ID before introducing
    # the FN ID...
    for overlapping_seg_id in np.unique(overlapping_seg_ids):
        # we have no FPs so this seg ID has already been matched to something. We need to find what it is
        overlapping_node_id = t_label_to_node_id[(merge_t, overlapping_seg_id)]
        matched_gt_id = pred_to_gt[overlapping_node_id]
        matched_gt_label = gt_graph.nodes[matched_gt_id]['segmentation_id']
        matched_gt_mask = fn_frame == matched_gt_label

        # the replaced segs keep their original IDs
        seg[merge_t] = np.where(merge_frame == overlapping_seg_id, 0, merge_frame)
        seg[merge_t][matched_gt_mask] = overlapping_seg_id
    
    # now that we've replaced overlapping IDs, we copy over the FN mask
    seg[merge_t][fn_mask] = new_id

def update_and_save_segmentations(merge_df, summary_df, sol_root, out_seg_dir, as_zarr=False):
    merge_df['merge_t'] = -1
    merge_df['new_seg_id'] = -1

    merges_with_fn_succ = merge_df[merge_df.parent_has_fn_succ == True]
    for name, group in tqdm(merges_with_fn_succ.groupby('ds_name')):
        single_name, seq = name.split('_')
        if os.path.exists(os.path.join(out_seg_dir, single_name, f'{seq}_ERR_SEG')):
            print(f"Skipping dataset {name} as output directory already exists.")
            continue
        # load all the stuff
        summary_info = summary_df[summary_df.ds_name == name]
        seg_path = summary_info.seg_path.values[0]
        gt_seg_path = summary_info.tra_gt_path.values[0]
        sol_graph, gt_graph, seg, gt, pred_to_gt = load_sol_info(name, sol_root, seg_path, gt_seg_path, as_zarr=as_zarr)

        new_id = np.max(seg) + 1
        t_label_to_node_id = {
            (node_info['t'], node_info['label']) : node_id 
            for node_id, node_info in sol_graph.nodes(data=True)
        }
        for row in group.itertuples():
            fn_succ = row.fn_succ
            fn_info = gt_graph.nodes[fn_succ]
            fn_t = fn_info['t']
            fn_label = fn_info['segmentation_id']

            merge_v = row.merge_node
            merge_info = sol_graph.nodes[merge_v]
            merge_t = merge_info['t']

            if fn_t != merge_t:
                continue

            fn_frame = gt[fn_t]
            merge_frame = seg[merge_t]
            # if not fn overlaps, just copy over the fn_label mask with a new seg ID
            if not row.fn_overlaps:
                fn_mask = fn_frame == fn_label
                merge_frame[fn_mask] = new_id
            else:
                replace_fn_and_overlapping_mask(
                    fn_frame,
                    seg,
                    merge_t,
                    fn_label,
                    t_label_to_node_id,
                    new_id,
                    pred_to_gt,
                    gt_graph
                )

            merge_df.loc[row.Index, 'merge_t'] = merge_t
            merge_df.loc[row.Index, 'new_seg_id'] = new_id
            new_id += 1
        # once dataset is finished, save out new segmentation
        name, seq = name.split('_')
        new_seg_path = os.path.join(out_seg_dir, name, f'{seq}_ERR_SEG')
        save_tiff_frames(new_seg_path, seg)
    return merge_df

# step 2: for all merges with an fn successor, create the updated segmentation
# by introducing the merges, using the merge file from step 1
# as source of truth

# step 3: solve all updated segmentations allowing merges 

# step 4: evaluate all new solutions

if __name__ == '__main__':
    GT_DIR = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/'
    OUT_SEG_ROOT = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/merge_resolution/'
    
    sol_root = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/merge_resolution/merge_introduced_solved_183/'
    summary_pth = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/merge_resolution/merge_introduced_solved_183/summary.csv'
    summary_df = pd.read_csv(summary_pth)


    merge_df = get_basic_merge_df_for_sol_set(sol_root, as_zarr=True)
    merge_df = populate_merge_resolution_info_for_sol_set(merge_df, sol_root, as_zarr=True)
    merge_df = populate_overlap_info_for_sol_set(merge_df, summary_df, sol_root, as_zarr=True)
    merge_df.to_csv(os.path.join(sol_root, 'merge_info.csv'), index=False)

    solve_iter = 184
    while len(merge_df[merge_df.parent_has_fn_succ == True]):
        # generate updated segmentations 
        out_seg_root_iter = os.path.join(OUT_SEG_ROOT, f'merge_introduced_seg_{solve_iter}')
        out_sol_root_iter = os.path.join(OUT_SEG_ROOT, f'merge_introduced_solved_{solve_iter}')
        merge_df = update_and_save_segmentations(merge_df, summary_df, sol_root, out_seg_root_iter, as_zarr=True)
        merge_df.to_csv(os.path.join(sol_root, 'merge_info.csv'), index=False)

        # solve updated segmentations
        os.makedirs(out_sol_root_iter, exist_ok=True)
        out_csv_path = os.path.join(out_sol_root_iter, 'summary.csv')
        summary_df = generate_ctc_summary(out_seg_root_iter, out_csv_path, gt_dir=GT_DIR, use_err_seg=True)
        configs = ds_summary_to_configs(out_csv_path, out_sol_root_iter, div_constraint=False)
        for config in configs:
            config.run(compute_additional_features=True)
        # evaluate updated solutions
        for config in configs:
            config.evaluate()

        # update all merge info for next iter
        merge_df = get_basic_merge_df_for_sol_set(out_sol_root_iter, as_zarr=True)
        merge_df = populate_merge_resolution_info_for_sol_set(merge_df, out_sol_root_iter, as_zarr=True)
        merge_df = populate_overlap_info_for_sol_set(merge_df, summary_df, out_sol_root_iter, as_zarr=True)
        merge_df.to_csv(os.path.join(out_sol_root_iter, 'merge_info.csv'), index=False)

        sol_root = out_sol_root_iter
        solve_iter += 1
