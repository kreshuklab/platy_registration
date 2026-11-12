from pathlib import Path
import numpy as np
import argparse
from platy_reg.n5_utils import read_volume, write_volume, get_attrs
import logging
import sys

import tifffile

from platy_reg.vis import plot_three_slices

def main():
        parser = argparse.ArgumentParser(
        description="""Apply elastix registration transform to a new volume.
        """
        )
        parser.add_argument("input_n5", type=str, help="Path of the output n5")
        parser.add_argument("input_key", type=str, help="Key of ventral dataset")
        parser.add_argument("tiff_path", type=str, help="Path of the output n5")
        parser.add_argument("chan", type=int, help="Channel")
        args = parser.parse_args()

        img = read_volume(args.input_n5, args.input_key)
        if img.ndim == 4:
            img = img[args.chan, :, :, :]
        
        plot_three_slices(img, save_path="segm.png")
            
        tifffile.imwrite(args.tiff_path, img.astype(np.float32), imagej=True)        
        
        
if __name__=="__main__":
    main()
    
    
