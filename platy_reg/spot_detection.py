from skimage.feature import blob_dog, blob_log
import pandas as pd


def detect_spots(img):
    blobs_dog = blob_log(img, min_sigma=2, max_sigma=6, threshold=0.005)
    blob_table = pd.DataFrame(data=blobs_dog, index=range(len(blobs_dog)), columns=["z", "y", "x", "r"])
    # stats_columns = ["mean_intensity", "median_intensity", "max_intensity", "std_intensity"]
    # blob_stats = []
    # logging.info(f"Calculate blob stats")
    # for idx, blob in enumerate(blobs_dog):
    #     z, y, x, r = blob
    #     blob_stats.append(get_blob_stats(img, z, y, x, r))
    # blob_stats = pd.DataFrame(data=np.array(blob_stats), index=range(len(blobs_dog)), columns=stats_columns)
    # return pd.concat([blob_table, blob_stats], axis=1)
    return blob_table