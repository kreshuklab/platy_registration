from pathlib import Path
import numpy as np
import argparse
import os

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay

from scipy.ndimage import affine_transform


import logging
import sys



def transform_volume_affine(fixed_img: np.array, moving_img: np.array, affine_mat: np.array, order=0):
    registered_img = affine_transform(moving_img, np.linalg.inv(affine_mat), order=order, output_shape=fixed_img.shape, prefilter=False)
    return registered_img


def transform_affine_channelwise(fixed_img: np.array, moving_img: np.array, affine_mat: np.array, order=0):
    if moving_img.ndim == 4:
        result_img = []
        for chan in range(moving_img.shape[0]):
            output_img = transform_volume_affine(fixed_img, moving_img[chan, ...], affine_mat, order=order)
            result_img.append(output_img)
        result_img = np.array(result_img)
    else:
        result_img = transform_volume_affine(fixed_img, moving_img, affine_mat, order=order)
    return result_img


def main():
        parser = argparse.ArgumentParser(
        description="""Transform the volume to correspond to the sample orientation in the platy browser.
        """
        )
        parser.add_argument("em_n5", type=str, help="Path of the input n5 for EM segmentation in platy browser")
        parser.add_argument("em_key", type=str, help="Key of EM segmentation in platy browser")
        parser.add_argument("input_n5", type=str, help="Path of the input n5")
        parser.add_argument("input_key", type=str, help="Key of the volume to transform")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_key", type=str, help="Key to write aligned dataset to in the output n5")
        parser.add_argument("affine", type=str, help="Path to the transformation matrix to apply")
        parser.add_argument("order", type=int, help="Interpolation order")
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
        em_img_np = read_volume(args.em_n5, args.em_key)
        moving_img_np = read_volume(args.input_n5, args.input_key)
        affine_mat = np.loadtxt(args.affine)

        logging.info(f"Transform with interpolation order {args.order}")
        result_img = transform_affine_channelwise(em_img_np, moving_img_np, affine_mat, order=args.order)
        
        logging.info("Plot overlay")
        if result_img.ndim == 4:
            plot_overlay(em_img_np, result_img[-1, ...], log_dir / "transformed_overlay.png")
        else:
            plot_overlay(em_img_np, result_img, log_dir / "transformed_overlay.png")
        
        logging.info("Write results")

        attributes = dict(get_attrs(args.input_n5, args.input_key))
        
            
        if args.order == 0:
            logging.info("Converted image from dtype {result_img.dtype} to uint16, because interpolation order was 0")
            result_img = result_img.astype(np.uint16)
            
        if moving_img_np.ndim == 3:
            chunks = (128, 512, 512)
            attributes["resolution"] = get_attrs(args.em_n5, args.em_key)["resolution"]
            write_volume(args.output_n5, result_img, args.output_key, chunks=chunks, attrs=attributes)
        else:
            chunks = (128, 512, 512)
            attributes["resolution"] = get_attrs(args.em_n5, args.em_key)["resolution"]
            for chan in range(moving_img_np.shape[0]):
                chan_key = os.path.join(args.output_key, f"chan_{chan}")
                write_volume(args.output_n5, result_img[chan, ...], chan_key, chunks=chunks, attrs=attributes)
                
        
        
if __name__=="__main__":
    main()
    
    
