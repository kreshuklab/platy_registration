from pathlib import Path
import numpy as np
import argparse
from aicsimageio import readers

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay

from aicsimageio.aics_image import AICSImage
import logging
import sys

from platy_reg.preprocessing import percentile_norm
from platy_reg.elastix_utils import *

import itk


def elastix_rigid_alignment(fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, chan, log_dir):
    """
    Run rigid alignment of the ventral and dorsal datasets using elastix.
    """
    
    logging.info(f"Do rigid transform of unnormalized images")
    logging.info(f"Fixed resolution {fixed_resolution}")
    logging.info(f"Moving resolution {moving_resolution}")
    logging.info(f"Do rigid transform of unnormalized images")
    fixed_img = itk_scalar_img(fixed_img_np, fixed_resolution, ch=chan)
    moving_img = itk_scalar_img(moving_img_np, moving_resolution, ch=chan)
    
    plot_overlay(fixed_img_np[chan, ...], moving_img_np[chan, ...], log_dir / "intersample_overlay_before_alignment.png")
    
    parameter_map_paths = ["inter_sample_registration/ParameterMap_rigid_registration_MI.txt", "inter_sample_registration/ParameterMap_similarity_registration_corr.txt"]
    # parameter_map_paths = ["data_management/ParameterMap_rigid_registration_log.txt"]
    logging.info(f"Run rigid registration")
    result_image, result_transform_parameters = run_registration(fixed_img, moving_img, parameter_map_paths, str(log_dir), log_name="elastix_log_rigid.log")
    # serialize_parameter_object(result_transform_parameters, "ParameterMap_rigid_transform", log_dir)
    
    plot_overlay(fixed_img_np[chan, ...], itk_to_np_order(itk.GetArrayFromImage(result_image)), log_dir / "intersample_rigid_alignment.png")
    
    result_img_np = apply_transform_chanwise(result_transform_parameters, moving_img_np, moving_resolution)

    logging.info(f"Result image shape {result_img_np.shape}")
    
    return result_img_np


def main():
        parser = argparse.ArgumentParser(
        description="""Align moving fluorescence volume to fixed volume.
        """
        )

        parser.add_argument("fixed_n5", type=str, help="Path of the output n5")
        parser.add_argument("fixed_key", type=str, help="Key of ventral dataset")
        parser.add_argument("moving_n5", type=str, help="Path of the output n5")
        parser.add_argument("moving_key", type=str, help="Key of ventral dataset")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_key", type=str, help="Key to write aligned ventral dataset to in the output n5")
        parser.add_argument("dapi_chan", type=int, help="Channel to use for registration")
        parser.add_argument("log_path", type=str, help="Path to log file")
        args = parser.parse_args()
        
        log_path = Path(args.log_path)
        
        logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(args.log_path, mode="w"),
                                logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')        
        
        logging.info("Read image file")
        fixed_img_np = read_volume(args.fixed_n5, args.fixed_key)
        
        if args.fixed_n5 == args.moving_n5:
            logging.info("Same n5 for fixed and moving image, not doing registration")
            moving_img_np = fixed_img_np
        
        else:
            moving_img_np = read_volume(args.moving_n5, args.moving_key)
            
            
            fixed_img_np = fixed_img_np.astype(np.float32)
            moving_img_np = moving_img_np.astype(np.float32)
            
            logging.info("Start registration")
            
            
            
            moving_img_np = elastix_rigid_alignment(fixed_img_np,
                                                    get_attrs(args.fixed_n5, args.fixed_key)["resolution"],
                                                    moving_img_np,
                                                    get_attrs(args.moving_n5, args.moving_key)["resolution"],
                                                    args.dapi_chan,
                                                    log_path.parent)
        
        logging.info("Write results")
        attributes = get_attrs(args.moving_n5, args.moving_key)
        write_volume(args.output_n5, moving_img_np, args.output_key, chunks=(1, 128, 512, 512), attrs=attributes)        
        
        
if __name__=="__main__":
    main()
    
    
