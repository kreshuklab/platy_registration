"""
Sample script to detect spots on a given channel with a 3D model with Spotiflow and save the results as a CSV file.

Basic usage:
    python predict_3d.py --input /path/to/image.tif --channel 0

Assumes the input image format is CZYX or ZYX.

Output file name is inferred from the input image file as well as the channel index.
By default, will store results to the folder "../predictions".

If running into memory issues, consider reducing the number of tiles, either by argument (--n-tiles) when calling the script (number of tiles directly) 
or by changing the variable APPROX_DEFAULT_TILE_SIZE. The latter is only used if the number of tiles is not given.


For testing purposes, you can add the --debug flag to predict on a small crop of the image (without saving the output).
"""

import argparse
from pathlib import Path

import numpy as np
from skimage import io
from spotiflow import utils
from spotiflow.model import Spotiflow
from platy_reg.n5_utils import read_volume, get_attrs
import matplotlib.pyplot as plt
import pandas as pd

# APPROX_DEFAULT_TILE_SIZE = (48, 224, 224)
APPROX_DEFAULT_TILE_SIZE = (32, 64, 64)
APPROX_DEFAULT_TILE_SIZE = (64, 128, 64)
APPROX_DEFAULT_TILE_SIZE = (64, 128, 128)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_n5", type=Path, required=True)
    parser.add_argument("--input_n5_key", type=Path, required=True)
    parser.add_argument("--channel", type=int, required=True)
    parser.add_argument("--output-folder", type=Path, default=Path("../predictions"))
    parser.add_argument("--model", type=Path, required=False, default=Path("../models/smfish3d_finetuned"))
    parser.add_argument("--n-tiles", nargs=3, default=None, required=False)
    parser.add_argument("--min-distance", type=int, default=1, required=False)
    parser.add_argument("--prob-thresh", type=float, default=None, required=False)
    parser.add_argument("--subpix", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--silent", action="store_true", default=False)
    args = parser.parse_args()

    assert args.input_n5.exists(), f"Input file not found: {args.input}"
    assert args.model.exists(), f"Model not found: {args.model}"
    assert args.prob_thresh is None or 0. < args.prob_thresh < 1., f"If given, probability threshold must be a floating point number bigger than 0 and smaller than 1. Received: {args.prob_thresh}"
    
    print("Reading input data...")
    roi = np.s_[args.channel, ...]
    print(f"Key: {args.input_n5_key}")
    img = read_volume(args.input_n5, str(args.input_n5_key), roi)
    assert img.ndim in (3, 4), f"Input image must have 3 (ZYX) or 4 (CZYX) dimensions. Received: {img.ndim}"
    if img.ndim == 4:
        assert args.channel < img.shape[0], f"Given channel index ({args.channel}) is equal or larger to the number of channels ({img.shape[0]})"
        img = img[args.channel]
    else:
        print("Ignoring channel argument as input is (Z,Y,X)")
        

    if args.debug:
        print("Debug mode. Will predict on crop without saving.")
        start_z, start_y, start_x = tuple(np.array(img.shape)//2)
        step_z, step_y, step_x = APPROX_DEFAULT_TILE_SIZE
        img = img[start_z:start_z+step_z, start_y:start_y+step_y, start_x:start_x+step_x]
        n_tiles = (1,1,1)
    else:
        n_tiles = tuple(args.n_tiles) if args.n_tiles is not None else tuple(s//t for s, t in zip(img.shape, APPROX_DEFAULT_TILE_SIZE))
    

    print("Loading model...")
    model = Spotiflow.from_folder(args.model, map_location="auto")

    print(f"Image shape is: {img.shape}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    out_fname = args.output_folder/f"chan_{args.channel}_spots.csv"

    print("Predicting on volume...")
    spots, details = model.predict(
        img,
        subpix=args.subpix,
        prob_thresh=args.prob_thresh,
        n_tiles=n_tiles,
        device="auto",
        min_distance=args.min_distance,
        verbose=not args.silent,
    )

    print("Prob shape", details.prob.shape)
    print("Heatmap shape", details.heatmap.shape)
    
    plt.imshow(details.heatmap[details.heatmap.shape[0] // 2])
    plt.savefig(args.output_folder / f"heatmap.png", dpi=300)
    
    plt.figure()
    plt.hist(details.prob, bins=100)
    plt.savefig(args.output_folder / f"probabilities.png", dpi=300)
    
    print("Number of spots", len(spots))
    
    if not args.debug:
        utils.write_coords_csv(spots, out_fname)

    
    spots_df = pd.read_csv(out_fname)
    cur_res = get_attrs(args.input_n5, str(args.input_n5_key))["resolution"]
    print("Resolution at which spots were detected:", cur_res)
    em_res = [0.32, 0.32, 0.32]
    rescaling_factors = [c / f for f, c in zip(em_res, cur_res)]
    print("EM resolution", em_res)
    print("Rescaling to EM resolution", rescaling_factors)
    for s, col in zip(rescaling_factors, ("z", "y", "x")):
        spots_df[col] = spots_df[col] * s
    spots_df.to_csv(args.output_folder/f"chan_{args.channel}_spots_em_res.csv", index=False)

    
    
    spots_df = pd.read_csv(out_fname)
    cur_res = get_attrs(args.input_n5, str(args.input_n5_key))["resolution"]
    print("Resolution at which spots were detected:", cur_res)
    plantseg_res = [0.129, 0.129, 0.129]
    rescaling_factors = [c / f for f, c in zip(plantseg_res, cur_res)]
    print("Plantseg resolution", plantseg_res)
    print("Rescaling to plantseg resolution", rescaling_factors)
    for s, col in zip(rescaling_factors, ("z", "y", "x")):
        spots_df[col] = spots_df[col] * s
    spots_df.to_csv(args.output_folder/f"chan_{args.channel}_spots_plantseg_res.csv", index=False)