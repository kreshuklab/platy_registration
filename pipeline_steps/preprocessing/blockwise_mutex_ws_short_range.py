import argparse
import h5py as h5

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from elf.segmentation.mutex_watershed import  mutex_watershed, blockwise_mutex_watershed

import logging
import sys

from platy_reg.n5_utils import read_volume, write_volume, get_attrs


def main():
    parser = argparse.ArgumentParser(
        description=""" Run blockwise mutex watershed on plantseg output 
        """
        )

    parser.add_argument("input_n5", type=str, help="Path of the output n5")
    parser.add_argument("input_key", type=str, help="Key of the input dataset")
    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_key", type=str, help="Key to write normalized dataset to in the output n5")
    parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
    parser.add_argument("log_path", type=str, help="Directory to store diagnostic plots etc")
    args = parser.parse_args()
    
    output_dir = Path(args.log_dir)
    output_dir.mkdir(exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(args.log_path, mode="w"),
                        logging.StreamHandler(sys.stdout)],
            datefmt='%Y-%m-%d %H:%M:%S')  
    # roi = np.s_[:, :, 512:1024, 512:1024]
    roi = np.s_[:]
    
    logging.info(f"Start reading predictions")
    # img_path = args.input_h5
    # img_file = h5.File(img_path, "r")
    # affinities = img_file[args.input_key][roi]
    # img_file.close()
    roi = np.s_[:]
    affinities = read_volume(args.input_n5, args.input_key, roi)
    
    logging.info(f"Affinities shape {affinities.shape}")
    
    # output_h5 = h5.File(args.output_h5, "a")
    
    z_pos = affinities.shape[1] // 2
    x_pos = affinities.shape[2] // 2

    plt.imshow(affinities[0, z_pos, :, :])
    plt.colorbar()
    plt.savefig(output_dir / "predicted_fg_xy.png", dpi=300)
    plt.close()
    plt.imshow(affinities[0, :, x_pos, :])
    plt.colorbar()
    plt.savefig(output_dir / "predicted_fg_xz.png", dpi=300)
    plt.close()
    
    
    plt.imshow(affinities[1, z_pos, :, :])
    plt.colorbar()
    plt.savefig(output_dir / "predicted_boundaries_xy.png", dpi=300)
    plt.close()
    plt.imshow(affinities[1, :, x_pos, :])
    plt.colorbar()
    plt.savefig(output_dir / "predicted_boundaries_xz.png", dpi=300)
    plt.close()
    
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-3, 0, 0], [0, -3, 0], [0, 0, -3]]
    
    fg = affinities[0] > 0.9
    # affinities = (affinities[2:, ...] > 0.2).astype(np.float32)
    affinities = affinities[2:-3, ...].astype(np.float32)
    block_shape = [256, 256, 256]
    
    logging.info(f"Start blockwise mutex watershed short range")
    instance_segm = blockwise_mutex_watershed(affinities, offsets, [1, 1, 1], block_shape, mask=fg, n_threads=32)
    logging.info(f"Blockwise watershed finished")

    instance_segm = instance_segm.astype(np.uint32)

    alpha = (instance_segm > 0).astype(np.float32)
    plt.imshow(instance_segm[z_pos, :, :], cmap="prism", alpha=alpha[z_pos, :, :])
    plt.colorbar()
    plt.savefig(output_dir / f"predicted_instance_segm_xy.png", dpi=300)
    plt.close()
    plt.imshow(instance_segm[:, x_pos, :], cmap="prism", alpha=alpha[:, x_pos, :])
    plt.colorbar()
    plt.savefig(output_dir / f"predicted_instance_segm_xz.png", dpi=300)
    plt.close()
    
    logging.info(f"Write results")
    # ds = output_h5.get(args.output_key) or output_h5.create_dataset(args.output_key, data=instance_segm)
    # ds[...] = instance_segm  
    # output_h5.close()
    
    
    attributes = get_attrs(args.input_n5, args.input_key)
    write_volume(args.output_n5, instance_segm, args.output_key, chunks=(1, 512, 512), attrs=attributes)
    

        

if __name__=="__main__":
    main()