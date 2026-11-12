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

from platy_reg.vis import plot_three_slices


def elastix_segm_rigid_alignment(fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, log_dir):
    """
    Run rigid alignment of the ventral and dorsal datasets using elastix.
    """
    
    logging.info(f"Do rigid transform of unnormalized images")
    logging.info(f"Fixed resolution {fixed_resolution}")
    logging.info(f"Moving resolution {moving_resolution}")
    logging.info(f"Do rigid transform of unnormalized images")
    fixed_img = itk_scalar_img(fixed_img_np, fixed_resolution)
    moving_img = itk_scalar_img(moving_img_np, moving_resolution)
    
    plot_overlay(fixed_img_np, moving_img_np, log_dir / "intersample_segm_overlay_before_alignment.png")
    
    parameter_map_paths = ["pipeline_steps/inter_sample_registration/ParameterMap_segm_rigid_registration_corr.txt"]
    # parameter_map_paths = ["data_management/ParameterMap_rigid_registration_log.txt"]
    logging.info(f"Run rigid registration")
    result_image, result_transform_parameters = run_registration(fixed_img, moving_img, parameter_map_paths, str(log_dir), log_name="elastix_log_rigid.log", set_threads=True)
    # serialize_parameter_object(result_transform_parameters, "ParameterMap_rigid_transform", log_dir)
    print(result_image.shape)
    # result_image = itk_to_np_order(itk.GetArrayFromImage(result_image))
    plot_overlay(fixed_img_np, itk.GetArrayFromImage(result_image), log_dir / "intersample_segm_rigid_alignment.png")
    
    # result_img_np = apply_transform_chanwise(result_transform_parameters, moving_img_np, moving_resolution)
    result_img_np = itk_to_np_order(itk.GetArrayFromImage(result_image))

    logging.info(f"Result image shape {result_img_np.shape}")
    
    return result_img_np


def main():
        parser = argparse.ArgumentParser(
        description="""Apply elastix registration transform to a new volume.
        """
        )
        parser.add_argument("moving_n5", type=str, help="Path of the output n5")
        parser.add_argument("moving_key", type=str, help="Key of ventral dataset")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_key", type=str, help="Key to write aligned ventral dataset to in the output n5")
        parser.add_argument("parameter_map", type=str, help="Path of the parameter file")
        parser.add_argument("interpolation_order", type=int, help="Interpolation order (0, 1, 2 or 3)")
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

        moving_img_np = read_volume(args.moving_n5, args.moving_key)
        moving_img_np = moving_img_np.astype(np.float32)
        moving_resolution = get_attrs(args.moving_n5, args.moving_key)["resolution"]
        
        if ("sample_processing" in args.moving_n5) or ("multiview_registration" in args.moving_n5):
            logging.info("Changing resolution to make nuclei sizes comparable")
            # Dirty hack for the sample size mismatch
            # For no apparent reason the size of the light samples seems to be ~20% larger than the EM
            moving_resolution = get_attrs(args.moving_n5, args.moving_key)["resolution"]
            # moving_resolution = [r * 0.9 for r in moving_resolution]
        
        if moving_img_np.ndim == 3:
            moving_img_np = moving_img_np[None, ...]
            chunks = (128, 512, 512)
        else:
            chunks = (1, 128, 512, 512)
        
        logging.info("Start transformation")

        logging.info(f"Read parameter file {args.parameter_map}")
        parameter_object = itk.ParameterObject.New()
        parameter_object.ReadParameterFile(args.parameter_map)
        parameter_object.SetParameter("FinalBSplineInterpolationOrder", str(args.interpolation_order))
        logging.info(parameter_object)
        result_img_np = apply_transform_chanwise(parameter_object, moving_img_np, moving_resolution)
            
        logging.info(f"Result image shape {result_img_np.shape}")
        
        logging.info("Plot transformed image")
        plot_three_slices(result_img_np[-1, ...], save_path=log_dir / "dapi_transformed.png")
        result_img_np = np.squeeze(result_img_np)
        
        logging.info("Write results")
        attributes = dict(get_attrs(args.moving_n5, args.moving_key))
        resolution = [float(res) for res in parameter_object.GetParameter(0, "Spacing")]
        attributes["resolution"] = resolution
        
        write_volume(args.output_n5, result_img_np, args.output_key, chunks=chunks, attrs=attributes)        
        
        
if __name__=="__main__":
    main()
    
    
