"""
helper files to download datasets needed for unsup object centric learning ..
"""

import os
import numpy as np
import tensorflow_datasets as tfds

# MOVi helper (default MOVi-E)
ds, ds_info = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", with_info=True)
train_iter = iter(tfds.as_numpy(ds["train"]))



