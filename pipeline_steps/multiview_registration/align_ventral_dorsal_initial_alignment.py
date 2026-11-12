from pathlib import Path
import numpy as np
import argparse

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay

import logging
import sys

from platy_reg.initial_orientation import orient_sample
from scipy.ndimage import rotate

import gc



def main():
        parser = argparse.ArgumentParser(
        description="""Preliminary orientation of the samples with body axis along Y, head closer to 0.
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the output n5")
        parser.add_argument("input_ventral_key", type=str, help="Key of ventral dataset")
        parser.add_argument("input_dorsal_key", type=str, help="Key of dorsal dataset")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_ventral_key", type=str, help="Key to write aligned ventral dataset to in the output n5")
        parser.add_argument("output_dorsal_key", type=str, help="Key to write aligned dorsal dataset to in the output n5")
        parser.add_argument("dapi_chan", type=int, help="DAPI channel")
        parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
        parser.add_argument("log_path", type=str, help="Path to log file")
        args = parser.parse_args()
        
        log_path = Path(args.log_path)
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(args.log_path, mode="w"),
                                logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')        
        
        logging.info("Read image file")
        ventral_img_np = read_volume(args.input_n5, args.input_ventral_key)
        dorsal_img_np = read_volume(args.input_n5, args.input_dorsal_key)        
        
        logging.info(" ")
        logging.info("Orient ventral")
        ventral_log_dir = log_dir / "ventral_init_alignment"
        ventral_log_dir.mkdir(exist_ok=True)
        ventral_init_aligned = orient_sample(ventral_img_np, args.dapi_chan, ventral_log_dir)
        
        del ventral_img_np
        n = gc.collect()
        logging.info(f"Number of unreachable objects collected by GC: {n}")
        
        logging.info(" ")
        logging.info("Orient dorsal")
        dorsal_log_dir = log_dir / "dorsal_init_alignment"
        dorsal_log_dir.mkdir(exist_ok=True)
        dorsal_init_aligned = orient_sample(dorsal_img_np, args.dapi_chan, dorsal_log_dir, dorsal=True)
        
        del dorsal_img_np
        n = gc.collect()
        logging.info(f"Number of unreachable objects collected by GC: {n}")
        
        plot_overlay(ventral_init_aligned[args.dapi_chan, ...], dorsal_init_aligned[args.dapi_chan, ...], log_dir / "ventral_dorsal_init_registered_overlay.png")
        logging.info("Write results")
        attributes = get_attrs(args.input_n5, args.input_dorsal_key)
        write_volume(args.output_n5, dorsal_init_aligned.astype(np.float32), args.output_dorsal_key, chunks=(1, 128, 512, 512), attrs=attributes)
        attributes = get_attrs(args.input_n5, args.input_ventral_key)
        write_volume(args.output_n5, ventral_init_aligned.astype(np.float32), args.output_ventral_key, chunks=(1, 128, 512, 512), attrs=attributes)     
        
        
if __name__=="__main__":
    main()
    
    
