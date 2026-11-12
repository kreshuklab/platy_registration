import z5py
from pathlib import Path, PurePath
import numpy as np
from typing import Tuple, Dict



def print_key_tree(f: z5py.File):
    print(f"Key structure of z5 file {f.filename}")
    f.visititems(lambda name, obj: print(name))


def read_volume(f: z5py.File, key: str, roi = np.s_[:]):
    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "r")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None
        
    ds.n_threads = 8
    print(f"Reading roi {roi} of volume {key} from {f.filename}")
    vol = ds[roi]
    print(f"Read volume with shape {vol.shape}, data type {vol.dtype}")
    
    return vol

def get_attrs(f: z5py.File, key: str):
    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None
    
    return ds.attrs


def create_ind(arr: np.array):
    stop = arr.shape
    start = [0] * len(stop)
    step = [1] * len(stop)
    return start, stop, step


def read_ind(start: Tuple, stop: Tuple, step: Tuple):
    ind_exp = tuple(slice(*slice_param) for slice_param in zip(start, stop, step))
    return ind_exp


def write_ind(ind_exp):
    start = tuple(sl.start for sl in ind_exp)
    stop = tuple(sl.stop for sl in ind_exp)
    step = tuple(sl.step for sl in ind_exp)
    return start, stop, step


def write_volume(f, arr: np.array, key, chunks=(1, 512, 512), attrs=None, n_threads=8):
    shape = arr.shape
    compression = "gzip"
    dtype = arr.dtype

    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    if key not in f.keys():
        print(f"Created dataset {key}")
        ds = f.create_dataset(key, shape=shape, compression=compression,
                                chunks=chunks, dtype=dtype)
    else:
        print(f"Overwriting {key}")
        ds = f[key]
    
    ds.n_threads = n_threads
    print(f"Writing array to {key}")
    ds[:] = arr

    if attrs is not None:
        print(f"Assigning attributes of the dataset")
        for key in attrs.keys():
            ds.attrs[key] = attrs[key]
