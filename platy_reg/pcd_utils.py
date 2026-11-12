
import numpy as np
from pathlib import Path
import open3d as o3d
import copy
import time
from probreg import cpd, callbacks
import logging
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from platy_reg.preprocessing import percentile_norm
from probreg.transformation import Transformation
from sklearn.metrics import jaccard_score, precision_score, accuracy_score, recall_score

import json


use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    asnumpy = cp.asnumpy
else:
    cp = np
    to_cpu = lambda x: x
    
    def asnumpy(x):
        return x


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
    


def create_KDTree(pcd: o3d.t.geometry.PointCloud):
    """Helper function to use newer tensor point cloud class with KDTree
    Args:
        pcd: _description_

    Returns:
        _description_
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd.point.positions.numpy().T)
    return pcd_tree


def draw_pcd_3D(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def plot_point_clouds(fixed_pcd: o3d.t.geometry.PointCloud, moving_pcd: o3d.t.geometry.PointCloud, fixed_col="cornflowerblue", moving_col="orangered", projection="xy", save_path=None, title=""):
    """"
    Overlay two point clouds. Parameters of plotting should be ok to visualize two full platy volumes
    """
    assert len(projection) == 2, f"Projection should be xy, yz or something like that of length 2, not {projection}"
    axis_order = {"x": 0, "y": 1, "z": 2}
    roi_x = np.s_[:, axis_order[projection[0]]]
    roi_y = np.s_[:, axis_order[projection[1]]]
    
    plt.figure(figsize=(10, 10))
    plt.cla()
    plt.axis("equal")
    fixed_np = fixed_pcd.point.positions.numpy()
    moving_np = moving_pcd.point.positions.numpy()
    # print(fixed_np)
    plt.title(title)
    plt.scatter(fixed_np[roi_x], fixed_np[roi_y], s=0.6, c=fixed_col, alpha=0.5, label="Fixed point cloud")
    plt.scatter(moving_np[roi_x], moving_np[roi_y], s=0.6, c=moving_col, alpha=0.5, label="Moving point cloud")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)


def match_points(
    fixed_pcd: o3d.t.geometry.PointCloud,
    moving_pcd: o3d.t.geometry.PointCloud,
    knn: int = 1,
    max_dist: float = None,
):
    """Create weight matrix to match two point clouds, such that fixed_mapped_signal = W (fxm) * moving_signal (mx1)

    Args:
        fixed_pcd: fixed point cloud
        moving_pcd: moving point cloud
        knn: number of nearest neighbors to consider
        max_dist: maximal distance to consider when looking for the neighbors
    Returns:
        dist_mat - matrix of distances to the found nearest neighbours of shape (fxm)
        W - row-normalized matrix of distances to the nearest neighbours of shape (fxm)
    """
    moving_tree = create_KDTree(moving_pcd)
    dist_mat = np.zeros(
        (len(fixed_pcd.point.positions), len(moving_pcd.point.positions)),
        dtype=np.float16,
    )

    for fixed_idx, point in enumerate(fixed_pcd.point.positions.numpy()):
        if fixed_idx % 1000 == 0:
            logging.info(f"{fixed_idx} points out of {len(fixed_pcd.point.positions)}")
        if max_dist:
            n, idx, distances = moving_tree.search_hybrid_vector_3d(
                point, max_dist, knn
            )
        else:
            n, idx, _ = moving_tree.search_knn_vector_3d(point, knn)
        if n > 0:
            dist_vec = np.zeros(len(moving_pcd.point.positions))
            for moving_idx, d in zip(idx, distances):
                dist_vec[moving_idx] = d
            dist_mat[fixed_idx, :] = dist_vec

    W = dist_mat / (dist_mat.sum(axis=1)[:, np.newaxis] + 1e-6)

    return dist_mat, W


def map_features(fixed_pcd: o3d.t.geometry.PointCloud, moving_pcd: o3d.t.geometry.PointCloud, max_dist=5, knn=3):
    logging.info(f"Assign points with max_dist {max_dist} and number of neighbours {knn}")
    dist_mat, W = match_points(fixed_pcd, moving_pcd, knn, max_dist ** 2)
    mapped_features = {}
    for feat_name, val in moving_pcd.point.items():
        if feat_name not in ["colors", "positions"]:
            mapped_features[feat_name] = np.dot(W, np.nan_to_num(moving_pcd.point[feat_name].numpy()))
    
    return dist_mat, W, mapped_features


def binarize_feature(x: np.array, thresh: float = 0.5):
    x_norm = percentile_norm(x, pmin=0, pmax=99)
    x_bin = (x > thresh).astype(np.uint16)
    return x_bin


def evaluate_registration(fixed_pcd: o3d.t.geometry.PointCloud, moving_pcd: o3d.t.geometry.PointCloud, max_dist=5, knn=3, thresh=0.2):
    logging.info(f"Find matching nuclei and assign registered signal for the fixed cloud")
    dist_mat, W, mapped_features = map_features(fixed_pcd, moving_pcd, max_dist=max_dist, knn=knn)
    
    metrics = {}
    
    for feat in mapped_features:
        metrics[feat] = {}
        
        fixed_feat = fixed_pcd.point[feat].numpy()
        moving_feat = mapped_features[feat]
        
        corr = np.corrcoef(moving_feat, fixed_feat, rowvar=False)[0, 1]
        metrics[feat]["corr"] = corr
        logging.info(f"Corr for {feat}: {corr}")
        
        fixed_bin = binarize_feature(fixed_feat, thresh=thresh)
        moving_bin = binarize_feature(moving_feat, thresh=thresh)
        accuracy = accuracy_score(fixed_bin, moving_bin)
        jaccard = jaccard_score(fixed_bin, moving_bin)
        precision = precision_score(fixed_bin, moving_bin)
        recall = recall_score(fixed_bin, moving_bin)
        
        metrics[feat]["accuracy"] = accuracy
        metrics[feat]["jaccard"] = jaccard
        metrics[feat]["precision"] = precision
        metrics[feat]["recall"] = recall
        
        logging.info(f"Accuracy for {feat}: {accuracy}")
        logging.info(f"Jaccard for {feat}: {jaccard}")
        logging.info(f"Precision for {feat}: {precision}")
        logging.info(f"Recall for {feat}: {recall}")
        
        
        
    n_fixed_matched = np.sum(W.sum(axis=1) > 0)
    n_moving_matched = np.sum(W.sum(axis=0) > 0)
    
    metrics["n_fixed_matched"] = int(n_fixed_matched)
    metrics["n_moving_matched"] = int(n_moving_matched)
    
    metrics["n_fixed_matched_ratio"] = n_fixed_matched / W.shape[0]
    metrics["n_moving_matched_ratio"] = n_moving_matched / W.shape[1]
        
    logging.info(f"Cells in the fixed image which got signal {n_fixed_matched}, ratio {n_fixed_matched / W.shape[0]}")
    logging.info(f"Cells in the moving image which got used {n_moving_matched}, ratio {n_moving_matched / W.shape[1]}")
    
    return dist_mat, W, mapped_features, metrics

        
        
class Plot3DCallback(object):
    """Display the 3D registration result of each iteration by making max projection along z axis (if axes were in XYZ order).

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, source_color="turquoise", target_color="violet", save_path: bool|str = False, keep_window: bool = True):
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self.save_path = save_path
        self._cnt = 0
        
        self.source_color = source_color
        self.target_color = target_color
        
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.plot(target[:, 0], target[:, 1], "g^", label="target", markersize=0.2)
        plt.plot(result[:, 0], result[:, 1], "bo", label="result", markersize=0.2)
        plt.legend()
        plt.draw()

    def __call__(self, transformation: Transformation) -> None:
        # print(transformation.w, transformation.g)
        # print("result: ", transformation.b, transformation.t)
        # print("result: ", transformation.rot, transformation.t)
        logging.info(f"Iteration {self._cnt}")
        self._result = transformation.transform(self._source)
        plt.figure(figsize=(10, 10))
        plt.cla()
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        # plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.title(f"Iteration {self._cnt}")
        plt.scatter(target[:, 0], target[:, 1], s=0.5, c=self.target_color, alpha=0.5, label="target")
        plt.scatter(result[:, 0], result[:, 1], s=0.5, c=self.source_color, alpha=0.5, label="result")
        plt.legend()
        if self.save_path:
            plt.savefig(self.save_path / "xy" / f"image_{self._cnt}.png", dpi=300)
            
            
        plt.figure(figsize=(10, 10))
        plt.cla()
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        # plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.title(f"Iteration {self._cnt}")
        plt.scatter(target[:, 0], target[:, 2], s=0.5, c=self.target_color, alpha=0.5, label="target")
        plt.scatter(result[:, 0], result[:, 2], s=0.5, c=self.source_color, alpha=0.5, label="result")
        plt.legend()
        if self.save_path:
            plt.savefig(self.save_path / "xz" / f"image_{self._cnt}.png", dpi=300)

        
        plt.figure(figsize=(10, 10))
        plt.cla()
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        # plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.title(f"Iteration {self._cnt}")
        plt.scatter(target[:, 1], target[:, 2], s=0.5, c=self.target_color, alpha=0.5, label="target")
        plt.scatter(result[:, 1], result[:, 2], s=0.5, c=self.source_color, alpha=0.5, label="result")
        plt.legend()
        if self.save_path:
            plt.savefig(self.save_path / "yz" / f"image_{self._cnt}.png", dpi=300)
        
        self._cnt += 1



def visualize_reg_results(fixed_pcd, moving_pcd, output_path, features=["gene1_mean"], params={}, prefix="nonrigid_reg"):
    # Assign signal
    dist_mat, W, mapped_features, metrics = evaluate_registration(fixed_pcd, moving_pcd, max_dist=10, knn=2, thresh=0.1)
    
    
    for key in params:
        metrics[key] = params[key]
    print(metrics)
    with open(output_path / f"{prefix}_metrics.json", "w") as fp:
        json.dump(metrics, fp)
    
    for feature in features:    
        # Plot fixed and moving point clouds with corresponding signals
        turquoise_cmap = sns.blend_palette(["paleturquoise", "darkslategray"], as_cmap=True)
        violet_cmap = sns.blend_palette(["pink", "purple"], as_cmap=True)
        fixed_color = [turquoise_cmap(val) for val in percentile_norm(fixed_pcd.point[feature].numpy(), pmin=0, pmax=99)]
        moving_color = [violet_cmap(val) for val in percentile_norm(moving_pcd.point[feature].numpy(), pmin=0, pmax=99)]
        
        for projection in ["xy", "yz", "xz"]:
            plot_point_clouds(
                fixed_pcd,
                moving_pcd,
                fixed_color,
                moving_color,
                title=f"{feature} intensity overlay {projection}",
                save_path=output_path / f"{prefix}_{feature}_{projection}.png",
                projection=projection,
            )        
            
        # Plot fixed point cloud with GT signal and shifted point cloud with assigned signal
        registered_color = [violet_cmap(val) for val in percentile_norm(mapped_features[feature], pmin=0, pmax=99)]
        shift = 1
        shifted_pcd = copy.deepcopy(fixed_pcd)
        shifted_pcd.point.positions = shifted_pcd.point.positions + shift
        
        for projection in ["xy", "yz", "xz"]:
            plot_point_clouds(
                fixed_pcd,
                shifted_pcd,
                fixed_color,
                registered_color,
                title=f"Gene1 mean intensity overlay {projection}",
                save_path=output_path / f"{prefix}_signal_assignment_{feature}_{projection}.png",
                projection=projection,
            )

        # Scatterplot of GT signal vs assigned signal
        plt.figure(figsize=(8,8))
        plt.title(feature)
        fixed_feat = fixed_pcd.point[feature].numpy()
        moving_feat = mapped_features[feature]
        plt.scatter(fixed_feat, moving_feat, alpha=0.3, s=1)
        plt.plot([fixed_feat.min(), fixed_feat.max()], [moving_feat.min(), moving_feat.max()], c="r")
        plt.xlabel("Ground truth signal")
        plt.ylabel("Assigned from registration")
        plt.savefig(output_path / f"{prefix}_signal_assignment_scatter_{feature}.png", dpi=300)