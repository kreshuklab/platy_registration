from pathlib import Path
import numpy as np
import argparse
from aicsimageio import readers

from platy_reg.n5_utils import read_volume, write_volume, get_attrs

from aicsimageio.aics_image import AICSImage
import logging
import sys

import re


def main():
        parser = argparse.ArgumentParser(
        description="""Convert fluorescence platynereis from LIF or VSI scene to n5 for further processing.
        """
        )

        parser.add_argument("raw_path", type=str, help="Path of the input LIF or VSI file")
        parser.add_argument("input_scene", type=str, help="Name of the scene in the LIF or VSI file")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_key", type=str, help="Key to write to in the output n5")
        parser.add_argument("log_path", type=str, help="Path of the output directory")
        parser.add_argument("--description", type=str, help="Optional description, if not given, scene name will be used")
        args = parser.parse_args()
        
        logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(args.log_path, mode="a"),
                                logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')        
        
        logging.info("Read image file")
        img = AICSImage(args.raw_path)
        logging.info(f"Reading image at {args.raw_path}")
        logging.info(f"Available scenes:")
        logging.info("\n".join(img.scenes))
        logging.info("")
        if len(img.scenes) == 2:
             logging.info(f"Since there is only one scene, setting current scene to {img.scenes[0]}")
             img.set_scene(img.scenes[0])
        else:
            logging.info(f"Setting current scene to {args.input_scene}")
            img.set_scene(args.input_scene)
        
        if args.description is None:
            description = args.input_scene
        else:
            description = args.description
            
        img_input = img.data
        # Workaround for confocal ROpsin samples
        
        wrong_scenes = ["Pl3_Dorsal--ROpsin1_647Ch1-DAPI_Ch2--40X1.3OIL-Px129nm-2048px",
                'Pl4_Ventral--ROpsin1_647Ch1-DAPI_Ch2--40X1.3OIL-Px129nm-2048px',
                'Pl4_Dorsal--ROpsin1_647Ch1-DAPI_Ch2--40X1.3OIL-Px129nm-2048px',
                'Pl2_Ventral--ROpsin1_647Ch1-DAPI_Ch2--40X1.3OIL-Px129nm-2048px',
                'Pl3_Ventral--ROpsin1_647Ch1-DAPI_Ch2--40X1.3OIL-Px129nm-2048px',
                'Pl1_Dorsal--ROpsin1_647Ch1-DAPI_Ch2--40X1.3OIL-Px129nm-2048px']
        
        if args.input_scene in wrong_scenes:
            logging.info(f"CHANGED CHANNEL ORDER")
            img_data = img_input[:, ::-1, :, :, :]
            logging.info(f"CHANGED CHANNEL ORDER")
            
        else:
            logging.info(f"CHANNEL ORDER IS FINE")
            img_data = img_input
        logging.info(f"Read data")
        img_data = np.squeeze(img_data)
        assert img_data.ndim == 4, f"Expected data in CZYX format, got {img_data.ndim} dimensions"
        
        attributes = {}
        attributes["description"] = description
        attributes["unit"] = "um"
        attributes["scene"] = args.input_scene
        attributes["resolution"] = list(img.physical_pixel_sizes)
        
        logging.info(f"Write volume")
        write_volume(args.output_n5, img_data.astype(np.float32), args.output_key, chunks=(1, 128, 512, 512), attrs=attributes)
        
if __name__=="__main__":
    main()