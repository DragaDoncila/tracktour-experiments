"""
The purpose of this script is to determine the number of unique merges
explored during all iterations of the merge resolution process.

The IDs of vertices can change from iteration to iteration, so we can't
rely on the vertex IDs alone to determine uniqueness.

We have stored the solution of each dataset at each iteration. This means
we can match it_{t+1} to it_{t} using traccuracy, and thereby ensure
that each vertex is matched correctly.

We start with n_merges = n_merges(it_0), which we compute trivially
by just counting the number of merges in each dataset's solution.

Then, for each iteration t:
    - We match it_{t} to it_{t-1} using traccuracy
    - For all merges in it_{t}:
        - If the matched vertex is not a merge in it_{t-1}
            - We increment n_merges by 1
"""

from collections import defaultdict
import os
from pathlib import Path
import networkx as nx
import pandas as pd

from traccuracy import TrackingGraph
from traccuracy.loaders import load_geff_data
from traccuracy.matchers import CTCMatcher

from tracktour import load_tiff_frames

pre_resolution_dir = '/home/ddon0001/PhD/experiments/scaled/no_div_constraint_err_seg'
post_introduce_dir = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/merge_resolution'

# not all datasets have merges, only consider those that do
ds_summary_df = pd.read_csv(f'{pre_resolution_dir}/summary.csv')

# ds_of_interest = ['BF-C2DL-MuSC_01']

ds_names = []
n_merges_pre = []
# for ds_name in ds_of_interest:
for ds_name in ds_summary_df.ds_name.unique():
    # load pred solution
    pred_path = f'{pre_resolution_dir}/{ds_name}/matched_solution.graphml'
    pred_graph = nx.read_graphml(pred_path, node_type=int)
    n_merges = 0
    for node in pred_graph.nodes:
        if pred_graph.in_degree(node) > 1:
            n_merges += 1
    if n_merges == 0:
        continue
    ds_names.append(ds_name)
    n_merges_pre.append(n_merges)

n_merges_df = pd.DataFrame({'ds_name': ds_names, 'n_merges_pre': n_merges_pre})
print(n_merges_df)

n_new_merges_per_iter = defaultdict(dict)
for ds_name in n_merges_df.ds_name.unique():
# for ds_name in ds_of_interest:
    ds, seq = ds_name.split('_')

    old_pred_graph = nx.read_graphml(f'{pre_resolution_dir}/{ds_name}/matched_solution.graphml', node_type=int)
    old_seg_pth = f'/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/{ds}/{seq}_ERR_SEG/'
    seg = load_tiff_frames(Path(old_seg_pth))
    old_pred_graph = TrackingGraph(old_pred_graph, segmentation=seg, label_key='label')

    for iter in range(0, 8):
        if iter == 0:
            sol_pth = f'{post_introduce_dir}/merge_introduced_solved/{ds_name}/matched_solution.zarr/pred.geff'
            seg_pth = f'/home/ddon0001/PhD/experiments/scaled/pre-thesis/merge_introduced_seg/{ds}/{seq}_ERR_SEG/'
        else:
            sol_pth = f'{post_introduce_dir}/merge_introduced_solved_{iter}/{ds_name}/matched_solution.zarr/pred.geff'
            seg_pth = f'{post_introduce_dir}/merge_introduced_seg_{iter}/{ds}/{seq}_ERR_SEG/'
        n_new_merges_iter = 0
        total_merges_iter = 0
        if os.path.exists(sol_pth):
            new_pred_graph = load_geff_data(Path(sol_pth), load_all_props=True, seg_property='label')
            new_pred_seg = load_tiff_frames(Path(seg_pth))
            new_pred_graph.segmentation = new_pred_seg
            new_pred_graph.label_key = 'label'

            frame = old_pred_graph.segmentation[1]
            print(frame.max())
            # match new_pred_graph to old_pred_graph using traccuracy
            matched = CTCMatcher().compute_mapping(old_pred_graph, new_pred_graph)
            # gt = old, pred = new
            old_to_new = {v[0]: v[1] for v in matched.mapping}
            new_to_old = {v[1]: v[0] for v in matched.mapping}
            for new_node in new_pred_graph.nodes:
                    # this is a merge in the new graph
                if new_pred_graph.graph.in_degree(new_node) > 1:
                    # if the new node existed in the old graph
                    if new_node in new_to_old:
                        old_node = new_to_old[new_node]
                        # but it wasn't previously a merge
                        if old_pred_graph.graph.in_degree(old_node) <= 1:
                            n_new_merges_iter += 1
                    else:
                        # this is a new node, so must be a new merge
                        n_new_merges_iter += 1
                    total_merges_iter += 1
            old_pred_graph = new_pred_graph
        
        n_new_merges_per_iter[ds_name][iter] = {
            'n_new_merges': n_new_merges_iter,
            'total_merges': total_merges_iter
        }

merge_sums = {
    ds: sum(info[i]['n_new_merges'] for i in info)
    for ds, info in n_new_merges_per_iter.items()
}

# add new column using map
n_merges_df["total_new_merges"] = n_merges_df["ds_name"].map(merge_sums)
n_merges_df["total_merges_inspected"] = n_merges_df["n_merges_pre"] + n_merges_df["total_new_merges"]
n_merges_df.to_csv('/home/ddon0001/PhD/experiments/scaled/pre-thesis/merge_resolution/num_unique_merges.csv', index=False)

