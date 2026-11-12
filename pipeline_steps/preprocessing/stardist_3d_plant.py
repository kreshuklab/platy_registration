from pathlib import Path
import z5py
from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.preprocessing import percentile_norm, zero_mean_unit_variance
from stardist.models import Config3D, StarDist3D, StarDistData3D
from skimage.transform import rescale
from skimage.color import label2rgb
from skimage import io
import matplotlib
import numpy as np
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import argparse
import logging
import sys
from platy_reg.vis import plot_overlay, plot_three_slices
import os



def resample_isotropic(img: np.array, resolution, order=0, anti_aliasing=False):
    
    final_res = 0.25
    
    scale_factors = [res / final_res for res in resolution]
    new_resolution = [res / scale for res, scale in zip(resolution, scale_factors)]
    logging.info(f"Resolution: {resolution}")
    logging.info(f"Scaling factors: {scale_factors}")
    logging.info(f"New resolution: {new_resolution}")
    logging.info(f"Image shape: {img.shape}, {img.dtype}")
    isotropic_image = rescale(img, scale_factors, order=order, mode="constant", anti_aliasing=anti_aliasing)
    logging.info(f"Shape of rescaled image: {isotropic_image.shape}")
    return isotropic_image, new_resolution


def generate_new_attributes(attrs, resolution):
    new_attrs = dict(attrs)
    new_attrs["resolution"] = resolution
    return new_attrs


def stardist3d_predict(model_basedir, model_name, input): 
    # Load model
    model = StarDist3D(None, name=model_name, basedir=model_basedir)

    # labels, details = model.predict_instances(input, n_tiles=(2, 8, 8), verbose=True)
    labels, details = model.predict_instances(input, n_tiles=(2, 8, 4), verbose=True)
    # labels, details = model.predict_instances(input, n_tiles=(2, 2, 2), verbose=True)
    logging.info(f"Labels {labels.shape}")
    # logging.info(details)

    return labels

    
def main():
    parser = argparse.ArgumentParser(
        description="""Run nuclei segmentation for single-channel DAPI-stained organoids using Stardist3D. Resample to be isotropic first.
        """
    )
    parser.add_argument("input_n5", type=str, help="Path of the input n5")
    parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_n5_key", type=str, help="Key of the prediction in the output n5")
    parser.add_argument("dapi_chan", type=int, help="Channel of the nuclei signal")
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

    roi = np.s_[args.dapi_chan, ...]
    
    
    # roi = np.s_[args.dapi_chan, 80:120, 500:800, 500:800]
    raw = read_volume(args.input_n5, args.input_n5_key, roi)
    logging.info(f"Image shape {raw.shape}")
    raw = zero_mean_unit_variance(percentile_norm(raw, 1, 99.6))

    attrs = get_attrs(args.input_n5, args.input_n5_key)
    resolution = attrs['resolution']
    logging.info(f"Initial resolution: {resolution}")
    
    # Resample to isotropic resolution
    logging.info(f"Rescale image to be isotropic with approximately [0.25, 0.25, 0.25] pixel size")
    isotropic_image, new_resolution = resample_isotropic(raw, resolution, order=2)
    prediction_attrs = generate_new_attributes(attrs, resolution)
    # write_volume(args.output_n5, isotropic_image, args.output_n5_key + "isotropic", attrs=new_attrs)

    # model_basedir = "/g/kreshuk/buglakova/projects/fibsem_registration/fluo_segmentation/stardist/models"
    model_basedir = "/g/kreshuk/buglakova/projects/platy_registration/pipeline_steps/preprocessing/stardist_models"
    model_name = "stardist_gpu2_100x100x100_1000epochs"
    

    prediction = stardist3d_predict(model_basedir, model_name, isotropic_image.astype(np.float32))
    
    logging.info(f"Resample segmentation back to the initial resolution: {resolution}")
    scale_factors = [isotropic_res / res for isotropic_res, res in zip(new_resolution, resolution)]
    rescaled_prediction = rescale(prediction, scale_factors, order=0, mode="constant", anti_aliasing=False)
    logging.info(f"Shape of rescaled segmentation: {rescaled_prediction.shape}, dtype: {rescaled_prediction.dtype}")
    prediction_attrs["description"] = "Nuclei segmentation with Stardist3D \
            after (1, 99.6) percentile normalization and zero mean unit variance normalization"
    rescaled_prediction = rescaled_prediction[0:raw.shape[0], 0:raw.shape[1], 0:raw.shape[2]]
    plot_three_slices(rescaled_prediction, save_path= log_dir / f"segmentation.png", cmap="prism", alpha=True)
    write_volume(args.output_n5, rescaled_prediction, args.output_n5_key, attrs=prediction_attrs, chunks=(128, 512, 512))


    plantseg_resolution = [0.129, 0.129, 0.129]
    logging.info(f"Resample segmentation to the plantseg: {plantseg_resolution}")
    scale_factors = [isotropic_res / res for isotropic_res, res in zip(new_resolution, plantseg_resolution)]
    rescaled_prediction = rescale(prediction, scale_factors, order=0, mode="constant", anti_aliasing=False)
    logging.info(f"Shape of rescaled segmentation: {rescaled_prediction.shape}, dtype: {rescaled_prediction.dtype}")
    prediction_attrs = generate_new_attributes(attrs, plantseg_resolution)
    prediction_attrs["description"] = "Nuclei segmentation with Stardist3D \
            after (1, 99.6) percentile normalization and zero mean unit variance normalization"
    plot_three_slices(rescaled_prediction, save_path= log_dir / f"segmentation_plantseg_res.png", cmap="prism", alpha=True)
    write_volume(args.output_n5, rescaled_prediction, f"{args.output_n5_key}_plantseg_res", attrs=prediction_attrs, chunks=(128, 512, 512))

if __name__=="__main__":
    main()
    
    