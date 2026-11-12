from pathlib import Path
from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from skimage.transform import rescale
import numpy as np
import pandas as pd
import argparse
import logging
import sys
import matplotlib.pyplot as plt
from math import floor
from copy import deepcopy

from skimage.measure import regionprops_table


def get_gene_chan_df(raw: np.array, cell_regions: np.array, gene_chan: int, gene_spots: str, log_dir):
    if gene_chan >= 0:
        props = regionprops_table(cell_regions, intensity_image=raw[gene_chan, ...], properties=("label", "intensity_min", 'intensity_mean', 'intensity_max', 'image_intensity'))
        props['intensity_sum'] = [np.sum(dist) for dist in props['image_intensity']]
        props['intensity_median'] = [np.median(dist) for dist in props['image_intensity']]
        props.pop('image_intensity')
        gene1_props = pd.DataFrame(props)
    else:
        props = regionprops_table(cell_regions, intensity_image=np.zeros_like(raw[0, ...]), properties=("label", "intensity_min", 'intensity_mean', 'intensity_max', 'image_intensity'))
        props['intensity_sum'] = [np.sum(dist) for dist in props['image_intensity']]
        props['intensity_median'] = [np.median(dist) for dist in props['image_intensity']]
        props.pop('image_intensity')
        gene1_props = pd.DataFrame(props)
    
    gene_spots_path = Path(gene_spots)
    if gene_spots_path.is_file():
        logging.info(f"Gene chan {gene_chan} spots detection file {gene_spots} exists")
        spots_df = pd.read_csv(gene_spots_path)
        spot_counts = [cell_regions[int(spot["z"]) - 1, int(spot["y"]) - 1, int(spot["x"]) - 1] for _, spot in spots_df.dropna().iterrows()]
        spot_idx, spot_counts = np.unique(spot_counts, return_counts=True)
        spot_counts_df = pd.DataFrame(columns=["label", "counts"], data=np.array((spot_idx, spot_counts)).T)

        plt.figure()
        plt.scatter(spots_df["y"], spots_df["x"], alpha=0.1, s=3)
        plt.savefig(log_dir / f"{gene_spots_path.stem}_xy.png", dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(spots_df["z"], spots_df["x"], alpha=0.1, s=3)
        plt.savefig(log_dir / f"{gene_spots_path.stem}_zx.png", dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(spots_df["z"], spots_df["y"], alpha=0.1, s=3)
        plt.savefig(log_dir / f"{gene_spots_path.stem}_zy.png", dpi=300)
        plt.close()

    else:
        spot_counts_df = gene1_props[["label"]]
        spot_counts_df["counts"] = 0
    gene_df = pd.merge(gene1_props, spot_counts_df, on="label", how="left")
    gene_df.fillna(0, inplace=True)
    print(gene_df)
    return gene_df


    
def main():
    parser = argparse.ArgumentParser(
        description="""Assign gene signal to the cells, extract nuclei centers.
        """
    )
    parser.add_argument("input_n5", type=str, help="Path of the input n5")
    parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
    parser.add_argument("segm_n5", type=str, help="Path of the input n5")
    parser.add_argument("segm_n5_key", type=str, help="Key of the nuclei segmentation in the input n5")
    parser.add_argument("cell_regions_n5_key", type=str, help="Key of the nuclei segmentation in the input n5")
    parser.add_argument("output_path", type=str, help="File to write resulting table")
    parser.add_argument("dapi_chan", type=int, help="Channel of the nuclei signal")
    parser.add_argument("gene1_chan", type=int, help="Channel of the gene1")
    parser.add_argument("gene1_spots", type=str, help="Path to the spot detection table for gene1")
    parser.add_argument("gene2_chan", type=int, help="Channel of the gene2")
    parser.add_argument("gene2_spots", type=str, help="Path to the spot detection table for gene2")
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

    raw = read_volume(args.input_n5, args.input_n5_key)
    logging.info(f"Image shape {raw.shape}")
    
    nuclei_segm = read_volume(args.segm_n5, args.segm_n5_key)
    logging.info(f"Nuclei segmentation shape {nuclei_segm.shape}")
    cell_regions = read_volume(args.segm_n5, args.cell_regions_n5_key)
    logging.info(f"Cell regions shape {cell_regions.shape}")
    
    logging.info(f"Calculate nuclei and cell region props")
    nuclei_props = pd.DataFrame(regionprops_table(nuclei_segm, properties=("label", "area", 'centroid')))
    cell_props = pd.DataFrame(regionprops_table(cell_regions, properties=("label", "area", 'centroid')))
    merged_df = pd.merge(cell_props, nuclei_props, on="label", how="inner", suffixes=["_nuclei", "_cells"])
    
    logging.info(f"Calculate gene properties")

    gene1_df = get_gene_chan_df(raw, cell_regions, args.gene1_chan, args.gene1_spots, log_dir)
    gene2_df = get_gene_chan_df(raw, cell_regions, args.gene2_chan, args.gene2_spots, log_dir)
        
    merged_chan_df = pd.merge(gene1_df, gene2_df, on="label", how="inner", suffixes=["_gene1", "_gene2"])
    
    props = regionprops_table(nuclei_segm, intensity_image=raw[args.dapi_chan, ...], properties=("label", "intensity_min", 'intensity_mean', 'intensity_max'))
    dapi_prop = pd.DataFrame(props)
    
    merged_chan_df = pd.merge(merged_chan_df, dapi_prop, on="label", how="inner", suffixes=["", "_dapi"])
    
    logging.info(f"Merge dataframes")
    merged_df = pd.merge(merged_df, merged_chan_df, on="label", how="inner")
    
    merged_df.to_csv(args.output_path, index=False)
    print(merged_df)
    
    for gene_col in ["counts_gene1", "counts_gene2"]:
        logging.info(f"Plot expression distribution for {gene_col}")
        plt.figure()
        expr_sorted = np.sort(merged_df[gene_col])
        plt.plot(expr_sorted)
        plt.title(f"Cumulative distribution of expression levels for {gene_col}")
        plt.ylabel("Number of counts")
        plt.xlabel("Cell number")
        plt.savefig(log_dir / f"expr_{gene_col}.png", dpi=300)
        
        plt.figure()
        plt.plot(expr_sorted[expr_sorted > 0])
        plt.title(f"Cumulative distribution of expression levels for {gene_col}")
        plt.ylabel("Number of counts")
        plt.xlabel("Cell number")
        plt.savefig(log_dir / f"expr_nonzero_{gene_col}.png", dpi=300)

        logging.info(f"Plot spot count distribution for {gene_col}")
        segm_frame = deepcopy(cell_regions[nuclei_segm.shape[0] // 2, ...])
        label_ids = np.unique(segm_frame)[1:]
        for idx in label_ids:
            segm_frame[segm_frame == idx] = merged_df.loc[idx, gene_col]
        plt.figure()
        print(segm_frame[::10, ::10])
        print("MAX", segm_frame.max())
        
        plt.imshow(segm_frame, vmin=0)
        plt.colorbar()
        plt.imshow(raw[args.gene1_chan, nuclei_segm.shape[0] // 2, ...], alpha=raw[args.gene1_chan, nuclei_segm.shape[0] // 2, ...].clip(min=0, max=1), cmap="Reds")
        plt.savefig(log_dir / f"mapped_expression_{gene_col}.png", dpi=300)


    merged_df = merged_df.set_index("label", drop=False)
    for gene_col in ["intensity_sum_gene1", "intensity_sum_gene2"]:
        logging.info(f"Plot expression distribution for {gene_col}")
        segm_frame = deepcopy(cell_regions[nuclei_segm.shape[0] // 2, ...])
        label_ids = np.unique(segm_frame)[1:]
        for idx in label_ids:
            segm_frame[segm_frame == idx] = merged_df.loc[idx, gene_col]
        plt.figure()
        print(segm_frame[::10, ::10])
        print("MAX", segm_frame.max())
        
        plt.imshow(segm_frame, vmin=0)
        plt.colorbar()
        plt.imshow(raw[args.gene1_chan, nuclei_segm.shape[0] // 2, ...], alpha=raw[args.gene1_chan, nuclei_segm.shape[0] // 2, ...].clip(min=0, max=1), cmap="Reds")
        plt.savefig(log_dir / f"mapped_expression_{gene_col}.png", dpi=300)

        plt.figure()
        expr_sorted = np.sort(merged_df[gene_col])
        plt.plot(expr_sorted)
        plt.title(f"Cumulative distribution of expression levels for {gene_col}")
        plt.ylabel("Number of counts")
        plt.xlabel("Cell number")
        plt.savefig(log_dir / f"sum_expr_{gene_col}.png", dpi=300)

    

if __name__=="__main__":
    main()