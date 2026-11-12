import numpy as np
import logging
from skimage.transform import rescale
from typing import List

def zero_mean_unit_variance(img):
    return (img - np.mean(img)) / np.std(img)

def percentile_norm(img, pmin, pmax, eps=1e-10, channelwise=False):
        """Percentile normalization. Image is supposed to be C(Z)YX

        Args:
            img: _description_
            pmin: in percents
            pmax: in percents
            eps: _description_. Defaults to 1e-10.

        Returns:
            _description_
        """
        if channelwise:
            norm_img = np.zeros_like(img)
            for chan in range(norm_img.shape[0]):
                pmin_val = np.percentile(img[chan, ...], pmin)
                pmax_val = np.percentile(img[chan, ...], pmax)
                norm_img[chan, ...] = (img[chan, ...] - pmin_val) / (pmax_val - pmin_val + eps)
            return norm_img
        else:
            pmin = np.percentile(img, pmin)
            pmax = np.percentile(img, pmax)
            return (img - pmin) / (pmax - pmin + eps)
    
    
def convert_to_point_cloud(img: np.array, thresh=0):
    """
    Every voxel with val > thresh is a point in the point cloud
    """
    
    X = np.array(np.where(img >= thresh)).T
    
    return X


def resample_isotropic(img: np.array, resolution: List, final_res: float, order=0, anti_aliasing=False):
    """Resample image with given resolution to isotropic at resolution final_res

    Args:
        img: _description_
        resolution: _description_
        final_res: _description_
        order: _description_. Defaults to 0.
        anti_aliasing: _description_. Defaults to False.

    Returns:
        resampled image, new resolution vector
    """
    
    scale_factors = [res / final_res for res in resolution]
    new_resolution = [res / scale for res, scale in zip(resolution, scale_factors)]
    logging.info(f"Resolution: {resolution}")
    logging.info(f"Scaling factors: {scale_factors}")
    logging.info(f"New resolution: {new_resolution}")
    isotropic_image = rescale(img, scale_factors, order=order, mode="constant", anti_aliasing=anti_aliasing)
    logging.info(f"Shape of rescaled image: {isotropic_image.shape}")
    return isotropic_image, new_resolution