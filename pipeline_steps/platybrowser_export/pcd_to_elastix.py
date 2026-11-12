from pathlib import Path
import argparse


import logging
import sys

import open3d as o3d

def pcd_to_elastix(pcd_path, elastix_path):
    """_summary_

    Args:
        pcd_path: Point cloud in PCD format
        elastix_path: Point cloud in Elastix format, for example:
            point
            5
            2214.0 282.2 0.0
            2445.0 2013.0 0.0
            795.0 366.0 0.0
            153.0 609.0
            324.0 2322.0
    """
    
    pcd = o3d.t.io.read_point_cloud(pcd_path)
    points = pcd.point.positions.numpy()
    with open(elastix_path, "w") as f:
        f.write("point\n")
        f.write(f"{len(points)}\n")
        for x, y, z in points:
            f.write(f"{z} {x} {y}\n")


def main():
        parser = argparse.ArgumentParser(
        description="""Convert points stored in PCD to Elastix point set format
        """
        )
        parser.add_argument("pcd_path", type=str, help="Path of the output n5")
        parser.add_argument("elastix_path", type=str, help="Path of the output n5")
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
        
        logging.info(f"Convert point cloud {args.pcd_path}")

        pcd_to_elastix(args.pcd_path, args.elastix_path)
        
        logging.info("Done")     
        
        
if __name__=="__main__":
    main()
    
    
