from pathlib import Path
from platy_reg.n5_utils import read_volume
import argparse
import logging
import sys
import h5py
import numpy as np
from skimage.transform import rescale

    
def main():
    parser = argparse.ArgumentParser(
        description="""Save DAPI channel to h5 for plantseg.
        """
    )
    parser.add_argument("input_n5", type=str, help="Path of the input n5")
    parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
    parser.add_argument("output_h5", type=str, help="Path of the output n5")
    parser.add_argument("output_h5_key", type=str, help="Key of the prediction in the output n5")
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
    raw = read_volume(args.input_n5, args.input_n5_key, roi)
    logging.info(f"Image shape {raw.shape}")
    logging.info(f"Upsample by the factors [4, 1, 1]")
    raw = rescale(raw, [4, 1, 1], order=2)
    logging.info(f"Upscaled image shape {raw.shape}")
    logging.info(f"writing to {args.output_h5}")
    
    f = h5py.File(args.output_h5, "w")
    f.create_dataset(args.output_h5_key, data=raw)
    
    
if __name__=="__main__":
    main()