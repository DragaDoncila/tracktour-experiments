
import os

import pandas as pd


if __name__ == '__main__':
    solved_root = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/ducb_w_resolve_no_merges/'
    out_root = '/home/ddon0001/PhD/experiments/scaled/pre-thesis/ducb_w_resolve_no_merges_no_ws/'

    edge_csv_pths = [os.path.join(solved_root, f) for f in os.listdir(solved_root) if f.endswith('.csv')]
    for edge_csv_pth in edge_csv_pths:
        edge_csv = pd.read_csv(edge_csv_pth, dtype={'error_type': str}, keep_default_na=False)
        filename = os.path.basename(edge_csv_pth)
        out_pth = os.path.join(out_root, filename)
        edge_csv.loc[edge_csv["error_type"] == "WS", "solution_incorrect"] = True
        edge_csv['error_type'] = edge_csv['error_type'].replace("WS", "Correct")
        edge_csv.to_csv(out_pth, index=False)