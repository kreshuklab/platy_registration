from pathlib import Path
import numpy as np
import argparse
from platy_reg.n5_utils import read_volume, write_volume, get_attrs
import logging
import sys

import tifffile

from platy_reg.vis import plot_three_slices
from skimage.measure import regionprops_table
import pandas as pd


def main():
        parser = argparse.ArgumentParser(
        description="""Assign new coordinates according to the transformed nuclei segmentation.
        """
        )
        parser.add_argument("input_n5", type=str, help="Path of the input n5")
        parser.add_argument("input_key", type=str, help="Key of the transformed segmentation")
        parser.add_argument("old_gene_assignment", type=str, help="Path of the old gene signal assignment")
        parser.add_argument("new_gene_assignment", type=str, help="Path of the new gene assignment"),
        parser.add_argument("suffix", type=str, help="Suffix to add to new columns")
        args = parser.parse_args()

        nuclei_segm = read_volume(args.input_n5, args.input_key).astype(np.uint32)
        old_gene_assignment = pd.read_csv(args.old_gene_assignment)
        nuclei_props = pd.DataFrame(regionprops_table(nuclei_segm, properties=("label", "area", 'centroid', 'bbox')))
        print(nuclei_props)
        nuclei_props = nuclei_props.add_suffix(args.suffix)
        nuclei_props["label"] = nuclei_props["label" + args.suffix]
        
        merged_df = pd.merge(old_gene_assignment, nuclei_props, on="label", how="inner", suffixes=["", args.suffix])
        merged_df.to_csv(args.new_gene_assignment, index=False)
        
        
if __name__=="__main__":
    main()
    
    
