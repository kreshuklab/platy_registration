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
    
    logging.info(f"Create ITK objects for the images")
    logging.info(f"Fixed resolution {fixed_resolution}")
    logging.info(f"Moving resolution {moving_resolution}")
    fixed_img = itk_scalar_img(fixed_img_np, fixed_resolution, ch=chan)
    moving_img = itk_scalar_img(moving_img_np, moving_resolution, ch=chan)
    
    parameter_map_paths = ["pipeline_steps/multiview_registration/ParameterMap_rigid_registration_MI.txt", "pipeline_steps/multiview_registration/ParameterMap_rigid_registration_NC.txt"]
    logging.info(f"Run rigid registration")
    result_image, result_transform_parameters = run_registration(fixed_img, moving_img, parameter_map_paths, str(log_dir), log_name="elastix_log_rigid.log")
    
    plot_overlay(fixed_img_np[chan, ...], itk_to_np_order(itk.GetArrayFromImage(result_image)), log_dir / "ventral_dorsal_rigid_alignment.png")
    
    logging.info(f"Transform the image")
    result_img_np = apply_transform_chanwise(result_transform_parameters, moving_img_np, moving_resolution)

    logging.info(f"Result image shape {result_img_np.shape}")
    
    return fixed_img_np, result_img_np


def main():
        parser = argparse.ArgumentParser(
        description="""Roughly align dorsal and ventral projections using rigid transformation.
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
        
        # ventral_init_aligned = percentile_norm(ventral_img_np, pmin=1, pmax=99.9, channelwise=False).astype(np.float32)
        # dorsal_init_aligned = percentile_norm(dorsal_img_np, pmin=1, pmax=99.9, channelwise=False).astype(np.float32)
        
        ventral_init_aligned = ventral_img_np.astype(np.float32)
        dorsal_init_aligned = dorsal_img_np.astype(np.float32)
        
        logging.info("Start registration")
        
        ventral_elastix_aligned, dorsal_elastix_aligned = elastix_rigid_alignment(ventral_init_aligned,
                                                                                  get_attrs(args.input_n5, args.input_ventral_key)["resolution"],
                                                                                  dorsal_init_aligned,
                                                                                  get_attrs(args.input_n5, args.input_dorsal_key)["resolution"],
                                                                                  args.dapi_chan,
                                                                                  log_dir)
        logging.info("Plot overlay")
        plot_overlay(ventral_elastix_aligned[args.dapi_chan, ...], dorsal_elastix_aligned[args.dapi_chan, ...], log_dir / "ventral_dorsal_registered_rigid_overlay.png")
        
        logging.info("Write results")
        attributes = get_attrs(args.input_n5, args.input_dorsal_key)
        write_volume(args.output_n5, dorsal_elastix_aligned.astype(np.float32), args.output_dorsal_key, chunks=(1, 128, 512, 512), attrs=attributes)
        attributes = get_attrs(args.input_n5, args.input_ventral_key)
        write_volume(args.output_n5, ventral_elastix_aligned.astype(np.float32), args.output_ventral_key, chunks=(1, 128, 512, 512), attrs=attributes)        
        
        
if __name__=="__main__":
    main()
    
    
