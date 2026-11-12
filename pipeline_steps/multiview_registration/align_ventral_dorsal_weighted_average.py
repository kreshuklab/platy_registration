from pathlib import Path
import numpy as np
import argparse

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay, plot_three_slices

import matplotlib.pyplot as plt
import logging
import sys
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.transform import rescale
import gc


def main():
        parser = argparse.ArgumentParser(
        description="""Average aligned ventral and dorsal volumes.
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the output n5")
        parser.add_argument("input_ventral_key", type=str, help="Key of ventral dataset")
        parser.add_argument("input_dorsal_key", type=str, help="Key of dorsal dataset")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_average_key", type=str, help="Key to write averaged dataset to in the output n5")
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

        logging.info("Read image files")
        ventral_img_np = read_volume(args.input_n5, args.input_ventral_key)
        dorsal_img_np = read_volume(args.input_n5, args.input_dorsal_key)
        
        dorsal_img_np = np.clip(dorsal_img_np, a_min=0, a_max=None)
        ventral_img_np = np.clip(ventral_img_np, a_min=0, a_max=None)

        logging.info("Plot image difference")
        plot_three_slices((ventral_img_np - dorsal_img_np)[args.dapi_chan, ...], log_dir / "ventral_dorsal_registered_difference.png", x_pos=None, y_pos=None, z_pos=None, cmap="bwr", max_pos=False)

        
        attributes = dict(get_attrs(args.input_n5, args.input_dorsal_key))
        cur_res = attributes["resolution"]
        logging.info(f"Sample's resolution: {cur_res}")
        
        logging.info("Calculate weights")
        ventral_weights = gaussian(ventral_img_np[args.dapi_chan, ::10, ::10, ::10], sigma=(1, 2, 2))
        dorsal_weights = gaussian(dorsal_img_np[args.dapi_chan, ::10, ::10, ::10], sigma=(1, 2, 2))
        
        logging.info("Plot downsampled image weights")
        plot_three_slices(ventral_weights, log_dir / "ventral_dorsal_ventral_weights_downscaled.png", x_pos=None, y_pos=None, z_pos=None, max_pos=False)
        plot_three_slices(dorsal_weights, log_dir / "ventral_dorsal_dorsal_weights_downscaled.png", x_pos=None, y_pos=None, z_pos=None, max_pos=False)
        
        ventral_weights = resize(ventral_weights, ventral_img_np[args.dapi_chan].shape, order=1)
        dorsal_weights = resize(dorsal_weights, ventral_img_np[args.dapi_chan].shape, order=1)
        
        diff_weights = (ventral_weights >= dorsal_weights).astype(np.float32)
        diff_weights = gaussian(diff_weights, sigma=(2, 4, 4))
        
        ventral_weights = np.clip(diff_weights / diff_weights.max(), a_min=0, a_max=1)
        dorsal_weights = np.clip(1 - ventral_weights, a_min=0, a_max=1)
        
        logging.info(f"Ventral weights range {ventral_weights.min()} to {ventral_weights.max()}")
        logging.info(f"Dorsal weights range {dorsal_weights.min()} to {dorsal_weights.max()}")
        
        logging.info("Plot final image weights")
        plot_three_slices(ventral_weights, log_dir / "ventral_dorsal_ventral_weights.png", x_pos=None, y_pos=None, z_pos=None, max_pos=False)
        plot_three_slices(dorsal_weights, log_dir / "ventral_dorsal_dorsal_weights.png", x_pos=None, y_pos=None, z_pos=None, max_pos=False)
        plot_three_slices(ventral_weights - dorsal_weights, log_dir / "ventral_dorsal_ventral_weights_binary.png", cmap="bwr")
        
        ventral_weights = np.repeat(ventral_weights[np.newaxis, ...], ventral_img_np.shape[0], axis=0)
        dorsal_weights = np.repeat(dorsal_weights[np.newaxis, ...], dorsal_img_np.shape[0], axis=0)
        
        del diff_weights
        n = gc.collect()
        logging.info(f"Number of unreachable objects collected by GC: {n}")
                
        delta = 1e-7
        logging.info("Calculate average")
        averaged = (ventral_img_np * ventral_weights + dorsal_img_np * dorsal_weights) / (ventral_weights + dorsal_weights + delta)
        
        plot_three_slices(averaged[args.dapi_chan, ...], log_dir / "averaged.png", x_pos=None, y_pos=None, z_pos=None, max_pos=False)
        
        del ventral_img_np
        del ventral_weights
        del dorsal_img_np
        del dorsal_weights
        n = gc.collect()
        logging.info(f"Number of unreachable objects collected by GC: {n}")
        
        
        
        logging.info("Write results")
        write_volume(args.output_n5, averaged.astype(np.float32), args.output_average_key, chunks=(1, 128, 512, 512), attrs=attributes)

        logging.info("Resample to the EM resolution for EM registration")
        em_res = [0.32, 0.32, 0.32]
        
        rescaling_factors = [c / f for f, c in zip(em_res, cur_res)]
        logging.info(f"Sample's resolution: {cur_res}")
        logging.info(f"Resolution of the EM volume: {em_res}")
        logging.info(f"Rescaling factors {rescaling_factors}")
        averaged_em_res = rescale(averaged, [1 , *rescaling_factors], order=2)
        attributes_em = attributes
        attributes_em["resolution"] = em_res
        write_volume(args.output_n5, averaged_em_res.astype(np.float32), f"{args.output_average_key}_em_res", chunks=(1, 128, 512, 512), attrs=attributes_em)  
        

        logging.info("Resample to the plantseg resolution for plantseg segmentation, but only DAPI")
        plantseg_res = [0.129, 0.129, 0.129]
        rescaling_factors = [c / f for f, c in zip(plantseg_res, cur_res)]
        logging.info(f"Sample's resolution: {cur_res}")
        logging.info(f"Resolution at which plantseg model was trained: {plantseg_res}")
        logging.info(f"Rescaling factors {rescaling_factors}")
        averaged_plantseg_res = rescale(averaged[args.dapi_chan, ...], rescaling_factors, order=2)
        attributes_plantseg = attributes
        attributes_plantseg["resolution"] = plantseg_res
        write_volume(args.output_n5, averaged_plantseg_res.astype(np.float32), f"{args.output_average_key}_plantseg_res", chunks=(256, 256, 256), attrs=attributes_plantseg)

        
if __name__=="__main__":
    main()
    
    
