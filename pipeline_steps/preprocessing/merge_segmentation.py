import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


from skimage.transform import resize


from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
import logging
import sys
from platy_reg.n5_utils import read_volume, write_volume

from skimage.segmentation import find_boundaries
from skimage.transform import rescale
import gc


def plot_segmentation(segm, output_dir, z_pos, x_pos, name):
    instance_boundaries = find_boundaries(segm)
    alpha_boundaries = (instance_boundaries > 0).astype(np.float32)
    alpha = (segm > 0).astype(np.float32)
    plt.imshow(segm[z_pos, :, :], cmap="prism", alpha=alpha[z_pos, :, :])
    plt.imshow(instance_boundaries[z_pos, :, :], cmap="Greys", alpha=alpha_boundaries[z_pos, :, :])
    plt.colorbar()
    plt.savefig(output_dir / f"{name}_xy.png", dpi=300)
    plt.close()
    plt.imshow(segm[:, x_pos, :], cmap="prism", alpha=alpha[:, x_pos, :])
    plt.imshow(instance_boundaries[:, x_pos, :], cmap="Greys", alpha=alpha_boundaries[:, x_pos, :])
    plt.colorbar()
    plt.savefig(output_dir / f"{name}_xz.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=""" Use a heuristic to combine mutex watershed and stardist predictions
        """
        )
    parser.add_argument("stardist_n5_path", type=str, help="Path of the stardist prediction n5")
    parser.add_argument("stardist_key", type=str, help="Key of the stardist prediction")
    parser.add_argument("plantseg_pred_n5_path", type=str, help="Path of the affinities prediction n5")
    parser.add_argument("plantseg_pred_key", type=str, help="Key of the affinities prediction")
    parser.add_argument("mutex_ws_n5_path", type=str, help="Path of the mutex watershed n5")
    parser.add_argument("mutex_ws_key", type=str, help="Key of the mutex watershed")
    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_key", type=str, help="Key to write normalized dataset to in the output n5")
    parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
    parser.add_argument("log_path", type=str, help="Directory to store diagnostic plots etc")
    parser.add_argument("--max_size", type=int, default=40000, required=False)
    parser.add_argument("--min_size", type=int, default=9000, required=False)
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
    
    logging.info("Read foreground mask and boundary predictions")
    roi = np.s_[1, ...]
    boundaries = read_volume(args.plantseg_pred_n5_path, args.plantseg_pred_key, roi)
    
    z_pos = boundaries.shape[0] // 2
    x_pos = boundaries.shape[1] // 2

    plt.imshow(boundaries[z_pos, :, :])
    plt.colorbar()
    plt.savefig(output_dir / "predicted_boundaries_xy.png", dpi=300)
    plt.close()
    plt.imshow(boundaries[:, x_pos, :])
    plt.colorbar()
    plt.savefig(output_dir / "predicted_boundaries_xz.png", dpi=300)
    plt.close()
    
    logging.info("Read mutex ws predictions")
    roi = np.s_[:]
    mutex_ws = read_volume(args.mutex_ws_n5_path, args.mutex_ws_key, roi)
    mutex_ws = resize(mutex_ws, boundaries.shape, order=0)

    # img_file.close()
    
    plot_segmentation(mutex_ws, output_dir, z_pos, x_pos, "mutex_ws")
    
    logging.info("Read stardist and resize to match the mutex_ws")
    # roi = np.s_[:, 512:1024, 512:1024]
    roi = np.s_[:]
    stardist = read_volume(args.stardist_n5_path, args.stardist_key, roi=roi)
    stardist = resize(stardist, boundaries.shape, order=0)
    stardist = remove_small_objects(stardist, min_size=args.min_size)
    
    plot_segmentation(stardist, output_dir, z_pos, x_pos, "stardist")
    

    props = regionprops_table(
    mutex_ws,
    properties=('label', 'area'),
    )   
    
    plt.figure()
    plt.hist(props["area"], bins=100, log=True);
    plt.title("Mutex watershed size distribution before filtering")
    plt.savefig(output_dir / "size_hist_input.png")
    plt.close()
    
    # mutex_filtered = remove_small_objects(mutex_ws.astype(np.int64), min_size=args.min_size)
    mutex_filtered = mutex_ws
    fg_mask = mutex_ws> 0
    
    props_filtered = regionprops_table(
    mutex_filtered,
    properties=('label', 'area'),
    )   
    
    plt.figure()
    plt.hist(props_filtered["area"], bins=100, log=True);
    plt.title(f"Mutex watershed size distribution min_size={args.min_size}")
    plt.savefig(output_dir / "size_hist_small.png", dpi=300)
    plt.close()
    
    
    large_objects = remove_small_objects(mutex_filtered, min_size=args.max_size)
    small_objects = mutex_filtered ^ large_objects
    
    del mutex_filtered

    gc.collect()
    
    plot_segmentation(small_objects, output_dir, z_pos, x_pos, "mutex_ws_ok")
    
    plot_segmentation(large_objects, output_dir, z_pos, x_pos, "mutex_ws_large")
    
    
    props_stardist = regionprops_table(
    stardist,
    properties=('label', 'centroid'),
    )   
    
    logging.info(str(props_stardist))

    
    stardist_centroids = np.zeros_like(stardist)
    for z, y, x, idx in zip(props_stardist["centroid-0"], props_stardist["centroid-1"], props_stardist["centroid-2"], props_stardist["label"]):
        stardist_centroids[int(z), int(y), int(x)] = idx
    
    seeds = stardist_centroids * (large_objects > 0)
    seeds = measure.label(
        seeds,
        background=0,
        return_num=False,
        connectivity=None,
    )
    
        # Calculate watershed
    logging.info("Calculate watershed")
    # ws = watershed(image=hmap)
    threshold = 0.4
    sigma_seeds = 2.0
    # block_shape = [256, 256, 256]
    block_shape = [128] * 3
    # halo = [64, 64, 64]
    halo = [32] * 3
    # ws, max_id = blockwise_two_pass_watershed(boundaries, block_shape, halo, seeds=seeds, threshold=threshold, sigma_seeds=sigma_seeds, n_threads=8, verbose=True)

    
    ws = watershed(boundaries.astype(np.float32), markers=seeds, mask=(large_objects > 0), compactness=1)
    ws = (ws + small_objects.max()) * (large_objects > 0)
    
    plot_segmentation(ws, output_dir, z_pos, x_pos, "ws_large")
    merged = ws + small_objects
    
    del ws

    gc.collect()

    logging.info("Calculate watershed to remove small objects")
    merged = remove_small_objects(merged.astype(np.int64), min_size=args.min_size)
    merged = watershed(boundaries.astype(np.float32), markers=merged, mask=fg_mask, compactness=1)
    
    merged = remove_small_objects(merged.astype(np.int64), min_size=200)
    
    merged = measure.label(
        merged,
        background=0,
        return_num=False,
        connectivity=None,
    )
    
    
    merged = merged.astype(np.uint32)
    
    plot_segmentation(merged, output_dir, z_pos, x_pos, "merged")
    
    props_filtered = regionprops_table(
     merged,
    properties=('label', 'area'),
    )   
    
    plt.figure()
    plt.hist(props_filtered["area"], bins=100, log=True);
    plt.title(f"Merged size distribution min_size={args.min_size}")
    plt.savefig(output_dir / "size_hist_merged.png", dpi=300)
    plt.close()
    
    logging.info("Write final segmentation")
    write_volume(args.output_n5, merged, f"{args.output_key}_plantseg_res", chunks=(1, 512, 512), attrs={"resolution": [0.129, 0.129, 0.129]})
    
    logging.info("Resample to the EM resolution for EM registration")
    em_res = [0.32, 0.32, 0.32]
    cur_res = [0.129, 0.129, 0.129]
    rescaling_factors = [c / f for f, c in zip(em_res, cur_res)]
    logging.info(f"Sample's resolution: {cur_res}")
    logging.info(f"Resolution of the EM volume: {em_res}")
    logging.info(f"Rescaling factors {rescaling_factors}")
    merged_em_res = rescale(merged, rescaling_factors, order=0)
    write_volume(args.output_n5, merged_em_res, f"{args.output_key}_em_res", chunks=(128, 512, 512), attrs={"resolution": [0.32, 0.32, 0.32]})  
    
    

if __name__=="__main__":
    main()