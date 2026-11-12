from pathlib import Path
from platy_reg.n5_utils import read_volume, write_volume, get_attrs
import numpy as np
import argparse
import logging
import sys
from platy_reg.vis import plot_overlay, plot_three_slices
import os
from plantseg.functionals.prediction.prediction import unet_prediction


    
def main():
    parser = argparse.ArgumentParser(
        description="""Run nuclei segmentation for single-channel DAPI-stained organoids using Stardist3D. Resample to be isotropic first.
        """
    )
    parser.add_argument("input_n5", type=str, help="Path of the input n5")
    parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_n5_key", type=str, help="Key of the prediction in the output n5")
    parser.add_argument("model_weights_path", type=str, help="Path to the .pytorch checkpoint of plantseg")
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

    roi = np.s_[...]
    
    
    # roi = np.s_[80:300, 500:800, 500:800]
    raw = read_volume(args.input_n5, args.input_n5_key, roi)

    # raw = raw[:, :1000, ...]
    logging.info(f"Image shape {raw.shape}")
    plot_three_slices(raw, save_path= log_dir / f"raw.png", cmap="Greys_r", alpha=False)

    logging.info(f"Run prediction")

    # prediction = unet_prediction(raw,
    #                              input_layout="ZYX",
    #                              model_id=None, 
    #                              # patch=[128, 128, 128], 
    #                              patch = [192, 256, 256],
    #                              patch_halo=[32, 32, 32],
    #                              model_name="affinities_step2_filtered",
    #                              config_path="pipeline_steps/preprocessing/plantseg_models/affinities_step2_filtered/config_train.yml", 
    #                              model_weights_path="pipeline_steps/preprocessing/plantseg_models/affinities_step2_filtered/best_checkpoint.pytorch")
    
    prediction = unet_prediction(raw,
                                 input_layout="ZYX",
                                 model_id=None, 
                                 patch=[128, 128, 128], 
                                 # patch = [192, 256, 256],
                                 patch_halo=[32, 32, 32],
                                 model_name="plantseg_affogato_affinities_2rois",
                                 config_path="pipeline_steps/preprocessing/plantseg_models/plantseg_affogato_affinities_2rois/config_train.yml", 
                                 model_weights_path="pipeline_steps/preprocessing/plantseg_models/plantseg_affogato_affinities_2rois/best_checkpoint.pytorch")
    
    
    
    logging.info(f"Prediction shape {prediction.shape}")
    plot_three_slices(prediction[0, ...], save_path= log_dir / f"fg.png", cmap="Greys_r", alpha=False)
    plot_three_slices(prediction[1, ...], save_path= log_dir / f"boundaries.png", cmap="Greys_r", alpha=False)
    plot_three_slices(prediction[-1, ...], save_path= log_dir / f"affinities.png", cmap="Greys_r", alpha=False)
    resolution = get_attrs(args.input_n5, args.input_n5_key)["resolution"]
    prediction_attrs = {"description": "Affinities predicted using the model trained on plants and platy GT", "resolution": resolution}

    write_volume(args.output_n5, prediction, args.output_n5_key, attrs=prediction_attrs, chunks=(1, 256, 256, 256), n_threads=8)

if __name__=="__main__":
    main()
    
    