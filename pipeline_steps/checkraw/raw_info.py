from pathlib import Path
import numpy as np
import argparse
from aicsimageio import readers

from aicsimageio.aics_image import AICSImage
import logging
import sys
from platy_reg.vis import plot_three_slices


def main():
        parser = argparse.ArgumentParser(
        description="""Print information about LIF or VSI file and create overview images.
        """
        )

        parser.add_argument("raw_path", type=str, help="Path of the input LIF or VSI")
        parser.add_argument("output_dir", type=str, help="Path of the output directory")
        parser.add_argument("log_path", type=str, help="Path of the output directory")
        args = parser.parse_args()
        output_dir = Path(args.output_dir)
        
        logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(args.log_path, mode="w"),
                                logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
        
        logging.info(f"Read image file")
        # img = AICSImage(args.raw_path, reader=readers.LifReader)
        assert (args.raw_path.endswith("lif") or args.raw_path.endswith("vsi")), "Image is not vsi or lif"
        img = AICSImage(args.raw_path)
        logging.info(f"Reading image at {args.raw_path}")
        logging.info(f"Available scenes: {img.scenes}")
        
        out_file = open(output_dir / "scenes.txt", "a")

        for scene_idx, cur_scene in enumerate(img.scenes):
            logging.info(f"Read scene: scene_{scene_idx} {cur_scene}")
            if cur_scene == 'macro image':
                break
            img.set_scene(cur_scene)
            img_data = img.data
            logging.info(f"Scene shape {img_data.shape}")
            logging.info(f"Data type {img_data.dtype}")
            out_file.write(cur_scene + "\n")
            assert img_data.ndim == 5, f"Expected 5-dimensional image, got {img_data.ndim}"
            for chan in range(img_data.shape[1]):
                plot_three_slices(img_data[0, chan, :, :, :], output_dir / f"scene_{scene_idx}_chan{chan}.png")

        out_file.close()
        
if __name__=="__main__":
    main()