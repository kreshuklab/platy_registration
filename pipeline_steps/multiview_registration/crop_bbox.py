from pathlib import Path
import numpy as np
import argparse
from scipy.ndimage import center_of_mass

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay, plot_three_slices

import logging
import sys

from skimage.filters import gaussian
import gc



def bbox_3D(img, margin=10):
        if isinstance(margin, int):
                margin = [margin] * img.ndim


        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return max(rmin - margin[0], 0), min(rmax + margin[0], img.shape[0]), max(cmin - margin[1], 0), min(cmax + margin[1], img.shape[1]), max(zmin - margin[2], 0), min(zmax + margin[2], img.shape[2])


def main():
        parser = argparse.ArgumentParser(
        description="""Crop the image to reduce size.
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the output n5")
        parser.add_argument("input_key", type=str, help="Key of input dataset")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_key", type=str, help="Key to write the cropped dataset")
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
        img = read_volume(args.input_n5, args.input_key)        
        resolution = get_attrs(args.input_n5, args.input_key)["resolution"]
        
        # Set crop parameters
        downsample_factor = 10
        margin_um = 15
        margin = np.rint([margin_um / res for res in resolution])
        margin_downscaled = (margin // 10) + 1
        
        
        logging.info(f"Margin in um: {margin_um}")
        logging.info(f"Margin in pixels: {margin}")
        logging.info(f"Margin in pixels for downscaled image: {margin_downscaled}")
        
        logging.info(" ")
        logging.info("Downsample and smooth image")
        
        downsampled = img[args.dapi_chan, ::10, ::10, ::10]
        smoothed = gaussian(downsampled, sigma=2)
        trsh = np.percentile(smoothed, 80)
        logging.info(f"Threshold for smoothed image at percentile 90: {trsh}")
        
        plot_overlay((smoothed > trsh).astype(np.float32), downsampled, log_dir / "smoothed_mask.png")
        bbox = bbox_3D(smoothed > trsh, margin=margin_downscaled)
        zmin, zmax, ymin, ymax, xmin, xmax = [int(coord * 10) for coord in bbox]
        logging.info(f"Bbox {[coord * 10 for coord in bbox]}")
        
        
        cropped_img = img[:, zmin:zmax, ymin:ymax, xmin:xmax]
        logging.info(f"Cropped image shape {cropped_img.shape}")
        plot_three_slices(cropped_img[args.dapi_chan], log_dir / "cropped.png")

        # max_y_size = 2000
        # max_x_size = 1400

        print("DAPI max projection shape", np.sum(cropped_img[args.dapi_chan, ...], axis=0).shape)
        y_c, x_c = center_of_mass(np.sum(cropped_img[args.dapi_chan, ...], axis=0))
        logging.info(f"Center of mass of DAPI channel: {y_c}, {x_c}")

        # At resolution 0.129
        # max_y_size = 1600
        # max_x_size = 1250

        max_y_size = 1250
        max_x_size = 980

        if ymax - ymin > max_y_size:
                y_c = int(y_c)
                cropped_img = cropped_img[:, :, y_c - min(700, y_c):y_c + 900, :]
        plot_three_slices(cropped_img[args.dapi_chan], log_dir / "cropped_limited_size_y.png")

        if xmax - xmin > max_x_size:
                x_c = int(x_c)
                logging.info(f"Cropping ROI in X: {x_c - min(750, x_c)}, {x_c + 750}")
                cropped_img = cropped_img[:, :, :, x_c - min(750, x_c):x_c + 750]
        logging.info(f"Final image shape {cropped_img.shape}")
        
        del img
        n = gc.collect()
        logging.info(f"Number of unreachable objects collected by GC: {n}")
        
        plot_three_slices(cropped_img[args.dapi_chan], log_dir / "cropped_limited_size_xy.png")
        logging.info("Write results")
        attributes = get_attrs(args.input_n5, args.input_key)
        write_volume(args.output_n5, cropped_img.astype(np.float32), args.output_key, chunks=(1, 128, 512, 512), attrs=attributes)     
        
        
if __name__=="__main__":
    main()
    
    
