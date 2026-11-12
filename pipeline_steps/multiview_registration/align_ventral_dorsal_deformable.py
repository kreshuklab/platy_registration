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
    fixed_img = itk_scalar_img(fixed_img_np, fixed_resolution, ch=chan)
    moving_img = itk_scalar_img(moving_img_np, moving_resolution, ch=chan)
    
    parameter_map_paths = ["pipeline_steps/sample_processing/ParameterMap_rigid_registration_log.txt", "pipeline_steps/sample_processing/ParameterMap_rigid_registration_nonorm.txt"]
    # parameter_map_paths = ["data_management/ParameterMap_rigid_registration_log.txt"]
    logging.info(f"Run rigid registration")
    result_image, result_transform_parameters = run_registration(fixed_img, moving_img, parameter_map_paths)
    # serialize_parameter_object(result_transform_parameters, "ParameterMap_rigid_transform", log_dir)
    
    plot_overlay(fixed_img_np[chan, ...], itk_to_np_order(itk.GetArrayFromImage(result_image)), log_dir / "ventral_dorsal_rigid_alignment.png")
    
    result_img_np = apply_transform_chanwise(result_transform_parameters, moving_img_np, moving_resolution)

    logging.info(f"Result image shape {result_img_np.shape}")
    
    return fixed_img_np, result_img_np


def elastix_deformable_alignment(fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, chan, log_dir, elastix_param, step_name=""):
    """
    Run deformable alignment of the ventral and dorsal datasets using elastix.
    """

    logging.info(f"Do deformable alignment")
    
    norm = True
    if norm:
        fixed_img = itk_scalar_img(np.log(1 + fixed_img_np), fixed_resolution, ch=chan)
        moving_img = itk_scalar_img(np.log(1 + moving_img_np), moving_resolution, ch=chan)
    
    else:
        fixed_img = itk_scalar_img(fixed_img_np, fixed_resolution, ch=chan)
        moving_img = itk_scalar_img(moving_img_np, moving_resolution, ch=chan)
    
    parameter_map_paths = [elastix_param]
    print("Start registration")
    log_name = f"elastix_log_deformable_{step_name}.log"
    result_image, result_transform_parameters = run_registration(fixed_img, moving_img, parameter_map_paths, str(log_dir), log_name=log_name)
    
    result_image_np_chan = itk_to_np_order(itk.GetArrayFromImage(result_image))
    plot_overlay(fixed_img_np[chan, ...], result_image_np_chan, log_dir / f"ventral_dorsal_deformable_alignment_{step_name}.png")
    
    # transformix_filter = deformable_transform_diagnostics(result_transform_parameters, moving_img, log_dir)
    
    logging.info(f"Apply transform to all channels")
    
    apply_transform = True
    if apply_transform:
        result_img_np = apply_transform_chanwise(result_transform_parameters, moving_img_np, moving_resolution)
    else:
        result_img_np = moving_img_np
        result_img_np[chan, ...] = result_image_np_chan

    logging.info(f"Result image shape {result_img_np.shape}")
    
    return fixed_img_np, result_img_np



def main():
        parser = argparse.ArgumentParser(
        description="""Deformable alignment of two views of one sample.
        """
        )
        parser.add_argument("step_name", type=str, help="Name of the alignment step")
        parser.add_argument("elastix_param", type=str, help="Elastix parameter file")
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

        
        ventral_elastix_aligned, dorsal_elastix_aligned = elastix_deformable_alignment(ventral_img_np,
                                                                                  get_attrs(args.input_n5, args.input_ventral_key)["resolution"],
                                                                                  dorsal_img_np,
                                                                                  get_attrs(args.input_n5, args.input_dorsal_key)["resolution"],
                                                                                  args.dapi_chan,
                                                                                  log_dir,
                                                                                  args.elastix_param,
                                                                                  args.step_name)
        logging.info("Plot overlay")
        # plot_overlay(ventral_elastix_aligned[args.dapi_chan, ...], dorsal_elastix_aligned[args.dapi_chan, ...], log_dir / "ventral_dorsal_registered_deformable_overlay.png")
        
        logging.info("Write results")
        attributes = get_attrs(args.input_n5, args.input_dorsal_key)
        write_volume(args.output_n5, dorsal_elastix_aligned.astype(np.float32), args.output_dorsal_key, chunks=(1, 128, 512, 512), attrs=attributes)
        attributes = get_attrs(args.input_n5, args.input_ventral_key)
        write_volume(args.output_n5, ventral_elastix_aligned.astype(np.float32), args.output_ventral_key, chunks=(1, 128, 512, 512), attrs=attributes)        
        
        
if __name__=="__main__":
    main()
    
    
