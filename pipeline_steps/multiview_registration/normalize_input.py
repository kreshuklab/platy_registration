from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay, plot_three_slices

import logging
import sys


def normalize_chan(img, min_val, max_val, eps=1e-10):
    norm_img = (img - min_val) / (max_val - min_val + eps)
    norm_img_clip = np.clip(norm_img, a_min=0, a_max=1).astype(np.float32)
    return norm_img_clip


def normalize_img(img, plot_dir):
    norm_img = np.zeros(img.shape, dtype=np.float32)
    for chan in range(img.shape[0]):
        logging.info(f"Channel {chan}")
        plt.figure()
        img_chan = img[chan, ...]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Histogram of channel {chan} before and after normalization')
        ax1.hist(img_chan[img_chan > 0].flatten(), bins=100, log=True)
        ax1.set(xlabel="Raw intensity", ylabel="N pixels")
        ax1.set_title(f"Channel {chan} median {np.median(img_chan[img_chan > 0])} mean {img_chan[img_chan > 0].mean():.3f}")
        
        # min_val = img_chan[img_chan > 0].mean()
        # min_val = img_chan.mean()
        min_val = 0
        # Take max quantile such that 100 pixels with the highest intensity have intensity 1
        # So generalized, yay
        max_quantile = 1 - 100 / img_chan.size
        logging.info(f"Max quantile including 100 pixels is {max_quantile}")
        max_val = np.quantile(img_chan, max_quantile)
        logging.info(f"Normalize between {min_val:.3f} and {max_val:.3f}")
        img_chan_norm = normalize_chan(img_chan, min_val, img_chan.max())
        
        ax2.hist(img_chan_norm[img_chan_norm > 0].flatten(), bins=100, log=True)
        ax2.set(xlabel="Normalized intensity", ylabel="N pixels")
        ax2.set_title(f"Channel {chan} median {np.median(img_chan_norm[img_chan_norm > 0]):.3f} mean {img_chan_norm[img_chan_norm > 0].mean():.3f}")
        plt.savefig(plot_dir / f"hist_chan_{chan}.png", dpi=300)
        
        plot_three_slices(img_chan_norm, save_path=plot_dir / f"overview_{chan}.png", cmap="Greys_r")
        norm_img[chan, ...] = img_chan_norm
    return norm_img



def main():
        parser = argparse.ArgumentParser(
        description=""" Do percentile normalization per channel, plot histograms and 
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the output n5")
        parser.add_argument("input_key", type=str, help="Key of the input dataset")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_key", type=str, help="Key to write normalized dataset to in the output n5")
        parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
        parser.add_argument("log_path", type=str, help="Path to log file")
        args = parser.parse_args()
        
        log_path = Path(args.log_path)
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(args.log_path, mode="a"),
                                logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')        
        
        logging.info("Read image file")
        input_img_np = read_volume(args.input_n5, args.input_key)
        
        logging.info("Start normalization")
        norm_img = normalize_img(input_img_np, log_dir)
        
        logging.info("Write results")
        attributes = get_attrs(args.input_n5, args.input_key)
        write_volume(args.output_n5, norm_img.astype(np.float32), args.output_key, chunks=(1, 128, 512, 512), attrs=attributes)      
        
        
if __name__=="__main__":
    main()
    
    
