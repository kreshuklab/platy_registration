from pathlib import Path
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay
from platy_reg.preprocessing import percentile_norm

import logging
import sys

import open3d as o3d
from typing import List
import seaborn as sns

try:
    import cupy as cp

    asnumpy = cp.asnumpy
except:

    def asnumpy(x):
        return x



def create_pcd(coords_df: pd.DataFrame, resolution: List):
    # Resolution: ZYX, physical dimensions of pixels
    # coords_df: centroid-0, centroid-1, centroid-2 is ZYX in pixels
    # Open3D expects XYZ in physical coordinates
    
    # Scale coordinates and create points
    center_coords = np.array([coords_df["centroid-2_rigid"] * resolution[2], coords_df["centroid-1_rigid"] * resolution[1], coords_df["centroid-0_rigid"] * resolution[0]]).T.astype(np.float64)
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(center_coords)
    
    # Assign gene signal
    # Type and reshape, because o3d.core.Tensor requires numpy array as input and storing to .pcd specifically wants attributes of shape [N, :]
    coords_df = coords_df.fillna(0)
    pcd.point.gene1_spots = o3d.core.Tensor(np.array(coords_df['counts_gene1'])[:, None])
    pcd.point.gene2_spots = o3d.core.Tensor(np.array(coords_df['counts_gene2'])[:, None])
    pcd.point.gene1_mean = o3d.core.Tensor(np.array(coords_df['intensity_mean_gene1'])[:, None])
    pcd.point.gene2_mean = o3d.core.Tensor(np.array(coords_df['intensity_mean_gene2'])[:, None])
    pcd.point.gene1_sum = o3d.core.Tensor(np.array(coords_df['intensity_sum_gene1'])[:, None])
    pcd.point.gene2_sum = o3d.core.Tensor(np.array(coords_df['intensity_sum_gene2'])[:, None])
    pcd.point.gene1_median = o3d.core.Tensor(np.array(coords_df['intensity_median_gene1'])[:, None])
    pcd.point.gene2_median = o3d.core.Tensor(np.array(coords_df['intensity_median_gene2'])[:, None])
    pcd.point.label = o3d.core.Tensor(np.array(coords_df['label'])[:, None])
    
    # Assign colors: red for gene1 and yellow for gene2
    # red_cmap = sns.light_palette("crimson", as_cmap=True)
    # yellow_cmap = sns.light_palette("gold", as_cmap=True)
    
    # coords_df = coords_df.fillna(0)
    # pcd.point.gene1_counts_red = o3d.core.Tensor([red_cmap(val) for val in percentile_norm(coords_df['counts_gene1'], pmin=0, pmax=98)])
    # pcd.point.gene1_intensity_red = o3d.core.Tensor([red_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene1'], pmin=0, pmax=98)])
    # pcd.point.gene2_counts_yellow = o3d.core.Tensor([yellow_cmap(val) for val in percentile_norm(coords_df['counts_gene2'], pmin=0, pmax=98)])
    # pcd.point.gene2_intensity_yellow = o3d.core.Tensor([yellow_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene2'], pmin=0, pmax=98)])
    
    # Assign colors: get a random color for both genes and overall for all points to plot overlays nicely
    rand_color_palette = sns.color_palette("husl", 30) # arbitrary number 30, because that's a maximum of how many samples I'll have
    rand_col_uniform = rand_color_palette[np.random.randint(len(rand_color_palette))]
    pcd.paint_uniform_color(rand_col_uniform)
    
    # rand_col_gene1 = rand_color_palette[np.random.randint(len(rand_color_palette))]
    # rand_cmap = sns.light_palette(rand_col_gene1, as_cmap=True)
    # pcd.point.gene1_counts_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['counts_gene1'], pmin=0, pmax=98)])
    # pcd.point.gene1_intensity_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene1'], pmin=0, pmax=98)])
    
    # rand_col_gene2 = rand_color_palette[np.random.randint(len(rand_color_palette))]
    # rand_cmap = sns.light_palette(rand_col_gene2, as_cmap=True)
    # pcd.point.gene2_counts_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['counts_gene2'], pmin=0, pmax=98)])
    # pcd.point.gene2_intensity_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene2'], pmin=0, pmax=98)])
    
    
    # Assign colors: always same color if this is a fixed image and same color if it's moving - to do later
    
    return pcd


def plot_pcd(pcd: o3d.geometry.PointCloud, save_path=None):
    coords = pcd.point.positions.numpy()
    color = pcd.point.colors.numpy()
    logging.info("Point coordinates of shape {coords.shape} and dtype {coords.dtype}")
    # plt.scatter(coords[:, 0], coords[:, 1], s=0.2, c=color)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    red_cmap = sns.light_palette("crimson", as_cmap=True)
    yellow_cmap = sns.light_palette("gold", as_cmap=True)
    gene1_counts_red = [red_cmap(val) for val in percentile_norm(pcd.point.gene1_spots.numpy(), pmin=0, pmax=99)]
    gene1_intensity_red = [red_cmap(val) for val in percentile_norm(pcd.point.gene1_mean.numpy(), pmin=0, pmax=98)]
    gene2_counts_yellow = [yellow_cmap(val) for val in percentile_norm(pcd.point.gene2_spots.numpy(), pmin=0, pmax=99)]
    
    # print(percentile_norm(pcd.point.gene2_spots.numpy(), pmin=0, pmax=99))
    gene2_intensity_yellow = [yellow_cmap(val) for val in percentile_norm(pcd.point.gene2_mean.numpy(), pmin=0, pmax=98)]
        
    axs[0, 0].scatter(coords[:, 0], coords[:, 1], s=1, c=gene1_intensity_red, label="Gene 1 intensity", alpha=0.5)
    axs[0, 1].scatter(coords[:, 0], coords[:, 1], s=1, c=gene1_counts_red, label="Gene 1 counts", alpha=0.5)
    axs[1, 0].scatter(coords[:, 0], coords[:, 1], s=1, c=gene2_intensity_yellow, label="Gene 2 intensity", alpha=0.5)
    axs[1, 1].scatter(coords[:, 0], coords[:, 1], s=1, c=gene2_counts_yellow, label="Gene 2 counts", alpha=0.5)
    
    for ax in axs.flat:
        ax.legend()
        ax.set_facecolor("black")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def main():
        parser = argparse.ArgumentParser(
        description="""Convert nuclei centroid coordinates to a point cloud. Only take nuclei with not-None cell region to filter out random points outside main body. Store gene signal as point properties.
        """
        )

        parser.add_argument("segm_n5", type=str, help="Path of the input n5")
        parser.add_argument("segm_key", type=str, help="Key of the corresponding segmentation dataset")
        parser.add_argument("gene_assignment", type=str, help="Path to the csv with nuclei coordinates and gene expression values")
        parser.add_argument("pcd_path", type=str, help="Path to output point cloud file. .pcd format allows to store attributes")
        parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
        parser.add_argument("log_path", type=str, help="Path to log file")
        args = parser.parse_args()
        
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(args.log_path, mode="w"),
                                logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')        
        
        logging.info("Read gene assignment csv")
        coords_df = pd.read_csv(args.gene_assignment)
        attributes = get_attrs(args.segm_n5, args.segm_key)
        resolution = attributes["resolution"]
        
        pcd = create_pcd(coords_df, resolution)
        logging.info("Resulting point cloud")
        logging.info(f"{pcd}")
        
        logging.info("Plot point cloud")
        plot_pcd(pcd, save_path=log_dir / "point_cloud_genes.png")
        
        logging.info(f"Saving point cloud to {args.pcd_path}")
        o3d.t.io.write_point_cloud(args.pcd_path, pcd, write_ascii=True)
        
        
if __name__=="__main__":
    main()
    
    
