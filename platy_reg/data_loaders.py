from pathlib import Path
import h5py
from tifffile import imread
import z5py
from platy_reg.n5_utils import print_key_tree, read_volume, get_attrs


def get_pilot_data_paths():
    data_dir = Path("/g/kreshuk/buglakova/data/platy_registration/pilot_data/")
    n5_paths = list(data_dir.glob("*n5"))
    return n5_paths

def load_pilot_data_raw():
    n5_paths = get_pilot_data_paths()
    raw_imgs = [read_volume(f_n5, "raw") for f_n5 in n5_paths]
    return raw_imgs

def get_pilot_data_resolution():
    n5_paths = get_pilot_data_paths()
    raw_resolutions = [get_attrs(f_n5, "raw")["resolution"] for f_n5 in n5_paths]
    return raw_resolutions

def load_pilot_data_raw_aligned():
    n5_paths = get_pilot_data_paths()
    raw_imgs = [read_volume(f_n5, "raw_pca_aligned") for f_n5 in n5_paths]
    return raw_imgs