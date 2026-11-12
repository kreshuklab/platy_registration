import numpy as np
from pathlib import Path
import open3d as o3d
import copy
import time
import logging
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from platy_reg.preprocessing import percentile_norm

from platy_reg.ilp import read_index_pairs
import pandas as pd



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="""Assign gene expression from light to EM and assign EM features to light.
        """
        )

    parser.add_argument("em_features", type=str, help="Fixed point cloud (to which to assign signal)")
    parser.add_argument("sample_features", type=str, help="Registered point cloud (from which to transfer signal)")
    parser.add_argument("matched_pairs", type=str, help="Table with features of the fixed point cloud")
    parser.add_argument("em_ordered_df", type=str, help="Table with features of the registered point cloud")
    parser.add_argument("sample_ordered_df", type=str, help="Table with features of the registered point cloud")
    parser.add_argument("log_path", type=str, help="Path to log file")
    parser.add_argument("--gene1_name", type=str, default="", help="Name of gene 1. If empty, will not write gene1")
    parser.add_argument("--gene2_name", type=str, default="", help="Name of gene 2. If empty, will not write gene2")

    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    
    logging.basicConfig(level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.FileHandler(args.log_path, mode="w"),
                            logging.StreamHandler(sys.stdout)],
                datefmt='%Y-%m-%d %H:%M:%S')        
    
    logging.info(f"Assign features to EM from {args.sample_features}")
    logging.info(f"Gene names {args.gene1_name} {args.gene2_name}")
    
    sample_df = pd.read_csv(args.sample_features)
    print(sample_df)
    em_df = pd.read_csv(args.em_features)
    print(em_df)
    matched_pairs = read_index_pairs(args.matched_pairs)
    # print(matched_pairs)

    # Set up tables
    em_labels  = em_df["label_nuclei"]
    sample_labels = sample_df["label"]

    mobie_em_df = pd.DataFrame()
    mobie_em_df["label_id"] = em_labels.astype(int)
    
    mobie_sample_df = sample_df.copy()
    mobie_sample_df["label_id"] = sample_labels.astype(int)

    mobie_em_df["sample_label"] = sample_df["label"]
    if args.gene1_name != "nan":
        mobie_em_df[f"{args.gene1_name}_spots"] = sample_df["gene1_spots"]
        mobie_em_df[f"{args.gene1_name}_sum"] = sample_df["gene1_sum"]
        mobie_em_df[f"{args.gene1_name}_mean"] = sample_df["gene1_mean"]
    if args.gene2_name != "nan":
        mobie_em_df[f"{args.gene2_name}_spots"] = sample_df["gene2_spots"]
        mobie_em_df[f"{args.gene2_name}_sum"] = sample_df["gene2_sum"]
        mobie_em_df[f"{args.gene2_name}_mean"] = sample_df["gene2_mean"]


    mobie_em_df = mobie_em_df.set_index("label_id", drop=True)
    mobie_em_df.index.name="label_id"
    mobie_em_df.to_csv(args.em_ordered_df, sep="\t")

    mobie_sample_df.set_index("label_id", drop=True)
    mobie_sample_df.index.name="label_id"
    mobie_sample_df.to_csv(args.sample_ordered_df, sep="\t")


        # # Assign from light to EM
    # for idx in range(len(em_df)):
    #     matched_indices = [int(p2) for p1, p2 in matched_pairs if p1 == idx]

    #     col = "label"
    #     aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #     mobie_em_df.loc[idx, f"label"] = aggregated_values[0]

    #     if args.gene1_name:
    #         col = "gene1_spots"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene1_name}_spots"] = int(np.sum(aggregated_values))

    #         col = "gene1_mean"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene1_name}_mean"] = np.mean(aggregated_values)

    #         col = "gene1_sum"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene1_name}_sum"] = np.sum(aggregated_values)

    #         col = "gene1_median"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene1_name}_median"] = np.mean(aggregated_values)

    #     if args.gene2_name:
    #         col = "gene2_spots"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene2_name}_spots"] = int(np.sum(aggregated_values))

    #         col = "gene2_mean"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene2_name}_mean"] = np.mean(aggregated_values)

    #         col = "gene2_sum"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene2_name}_sum"] = np.sum(aggregated_values)

    #         col = "gene2_median"
    #         aggregated_values = [sample_df.loc[i, col] for i in matched_indices]
    #         mobie_em_df.loc[idx, f"{args.gene2_name}_median"] = np.mean(aggregated_values)


    
    
    