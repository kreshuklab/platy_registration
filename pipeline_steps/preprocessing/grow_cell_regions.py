from pathlib import Path
from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from skimage.transform import rescale, resize
import numpy as np
import argparse
import logging
import sys
from platy_reg.vis import plot_overlay, plot_three_slices

from skimage.segmentation import watershed
from skimage.filters import gaussian
from scipy.ndimage import distance_transform_edt

from platy_reg.preprocessing import resample_isotropic
import z5py
from z5py.util import remove_dataset



def get_cell_regions(img, segm, resolution):
    resample_resolution = 0.5
    isotropic_image, new_resolution = resample_isotropic(img, resolution, resample_resolution, order=0)
    isotropic_segm = resize(segm, isotropic_image.shape, order=0, mode="constant", anti_aliasing=False)
    
    segm_edt = distance_transform_edt(isotropic_segm == 0, sampling=new_resolution)
    
    max_dist = 2
    # mask = gaussian(isotropic_image, sigma=12)
    mask_bin = (segm_edt < max_dist).astype(np.uint8)
    # mask_bin = (mask > np.quantile(mask, 0.7)).astype(np.uint8)
    segm_edt[segm_edt > max_dist] = 0
    
    labels = watershed(segm_edt, isotropic_segm, mask=mask_bin)
    
    rescaled_labels = resize(labels, img.shape, order=0, mode="constant", anti_aliasing=False)
    rescaled_mask_bin = resize(mask_bin, img.shape, order=0, mode="constant", anti_aliasing=False)
    
    return rescaled_mask_bin, rescaled_labels

    
def main():
    parser = argparse.ArgumentParser(
        description="""Calculate approximate cell regions by using distance transform and watershed. Filter segmentation based on the blurred DAPI intensity to remove "nuclei" in chitae and background.
        """
    )
    parser.add_argument("raw_n5", type=str, help="Path of the input n5")
    parser.add_argument("raw_n5_key", type=str, help="Key of the dataset in the input n5")
    parser.add_argument("segm_n5", type=str, help="Key of the nuclei segmentation in the input n5")
    parser.add_argument("segm_n5_key", type=str, help="Key of the nuclei segmentation in the input n5")
    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_n5_key", type=str, help="Key of the prediction in the output n5")
    parser.add_argument("mask_n5_key", type=str, help="Key of the prediction in the output n5")
    parser.add_argument("filtered_segm_n5_key", type=str, help="Key of the prediction in the output n5")
    parser.add_argument("dapi_chan", type=int, help="Channel of the nuclei signal")
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

    roi = np.s_[args.dapi_chan, ...]
    raw = read_volume(args.raw_n5, args.raw_n5_key, roi=roi)
    attributes = get_attrs(args.raw_n5, args.raw_n5_key)
    logging.info(f"Image shape {raw.shape}")
    
    segm = read_volume(args.segm_n5, args.segm_n5_key)
    segm_attributes = dict(get_attrs(args.segm_n5, args.segm_n5_key)).copy()
    print(segm_attributes)
    logging.info(f"Segmentation shape {segm.shape}")
    
    if segm.shape != raw.shape:
        logging.info(f"Segmentation and image are not the same size -> resize")
        segm = resize(segm, raw.shape, order=0, mode="constant", anti_aliasing=False)
        

    # attrs = get_attrs(args.segm_n5, args.segm_n5_key)
    resolution = attributes['resolution']
    logging.info(f"Initial resolution: {resolution}")
    
    # Resample to isotropic resolution
    logging.info(f"Calculate approximate cell regions")    

    mask, rescaled_labels = get_cell_regions(raw, segm, resolution)
    filtered_segm = segm * mask
    logging.info(f"Filtered segm dtype {filtered_segm.dtype}")   
    
    plot_overlay(rescaled_labels, segm, save_path= log_dir / f"cell_regions.png")
    
    plot_three_slices(mask, save_path= log_dir / f"mask.png")
    
    plot_three_slices(filtered_segm.astype(np.float32), save_path= log_dir / f"filtered_segmentation.png")

    write_volume(args.output_n5, rescaled_labels, args.output_n5_key, chunks=(1, 512, 512), attrs=attributes)
    write_volume(args.output_n5, mask.astype(np.uint16), args.mask_n5_key, chunks=(1, 512, 512), attrs=attributes)
    write_volume(args.output_n5, filtered_segm, args.filtered_segm_n5_key, chunks=(1, 512, 512), attrs=attributes)


if __name__=="__main__":
    main()