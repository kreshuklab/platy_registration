from pathlib import Path
import numpy as np
import argparse


from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay

import logging
import sys
from skimage.measure import label
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import pandas as pd
import numpy as np


def main():
        parser = argparse.ArgumentParser(
        description="""Heuristic segmentation of neuropil.
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the output n5")
        parser.add_argument("input_key", type=str, help="Key of ventral dataset")
        parser.add_argument("em_n5", type=str, help="Path of the output n5")
        parser.add_argument("em_key", type=str, help="Key of ventral dataset")
        parser.add_argument("output_key", type=str, help="Key to write aligned ventral dataset to in the output n5")
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
        
        logging.info("Read cell regions segmentation")
        cell_regions = (read_volume(args.input_n5, args.input_key) > 0).astype(np.uint8)
        logging.info("Read em neuropil segmentation") 
        em_neuropil = (read_volume(args.em_n5, args.em_key) > 0).astype(np.uint8)
        
        cell_regions_labels = label(1 - cell_regions)
        indices, counts = np.unique(cell_regions_labels * em_neuropil, return_counts=True)
        print(indices, counts)
        indices = indices[1:]
        counts = counts[1:]

        light_neuropil = (cell_regions_labels == indices[np.argmax(counts)]).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        roi = np.s_[350, :, :]
        plt.imshow(cell_regions[roi], cmap="Greys_r")
        plt.imshow(light_neuropil[roi], cmap="Reds", alpha=1 * light_neuropil[roi])
        plt.imshow(em_neuropil[roi], cmap="Blues", alpha=0.5 * em_neuropil[roi])
        plt.savefig(log_dir / "neuropil_segm.png", dpi=300)

        logging.info("Write results")
        attributes = dict(get_attrs(args.input_n5, args.input_key))        
        write_volume(args.input_n5, light_neuropil, args.output_key, chunks=(128, 512, 512), attrs=attributes)        
        
        boundary = find_boundaries(light_neuropil, mode="inner")
        z, y, x = np.nonzero(boundary)
        random_points = np.random.choice(len(z), size=3000, replace=False)
        df = pd.DataFrame({"z": z[random_points] * 0.32, "y": y[random_points] * 0.32, "x": x[random_points] * 0.32})
        df = df[["x", "y", "z"]]
        df.to_csv(log_dir / "neuropil_points.csv", index=False)
        
if __name__=="__main__":
    main()
    
    
