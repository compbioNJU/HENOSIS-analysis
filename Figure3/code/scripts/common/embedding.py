"""Load normalized cell embeddings for heart conservation analyses."""

import h5py
import numpy as np
from sklearn.preprocessing import normalize


def read_category(group):
    categories = group['categories'].asstr()[:]
    codes = group['codes'][:]
    values = np.full(len(codes), '', dtype=object)
    valid = codes >= 0
    values[valid] = categories[codes[valid]]
    return values.astype(str)


def load_embedding(path):
    with h5py.File(path) as h:
        species = read_category(h['obs/species'])
        labels = read_category(h['obs/Celltype'])
        embedding = np.asarray(h['obsm/X_HENOSIS'], dtype=np.float32)
    valid = (species != '') & (labels != '') & np.isfinite(embedding).all(axis=1)
    return species[valid], labels[valid], normalize(embedding[valid], axis=1)


