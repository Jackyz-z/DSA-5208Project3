"""Read and write the DTREEBIN dataset format used by the C++ programs.

Layout (little endian):
    char[8]  magic = b"DTREEBIN"
    uint64   n_rows
    uint32   n_features
    uint32   n_classes
    uint8    labels[n_rows]
    float32  values[n_features][n_rows]   (column-major)
"""

import struct

import numpy as np

MAGIC = b"DTREEBIN"
HEADER = struct.Struct("<8sQII")


def create(path, n_rows, n_features, n_classes):
    """Create an empty dataset file and return writable (labels, values) memmaps.

    values has shape (n_features, n_rows), i.e. one contiguous row per feature.
    """
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC, n_rows, n_features, n_classes))
        f.truncate(HEADER.size + n_rows + 4 * n_rows * n_features)
    labels = np.memmap(path, dtype=np.uint8, mode="r+", offset=HEADER.size, shape=(n_rows,))
    values = np.memmap(path, dtype="<f4", mode="r+", offset=HEADER.size + n_rows,
                       shape=(n_features, n_rows))
    return labels, values


def load(path):
    """Return (n_classes, labels, values) as read-only memmaps."""
    with open(path, "rb") as f:
        magic, n_rows, n_features, n_classes = HEADER.unpack(f.read(HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"{path} is not a DTREEBIN file")
    labels = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER.size, shape=(n_rows,))
    values = np.memmap(path, dtype="<f4", mode="r", offset=HEADER.size + n_rows,
                       shape=(n_features, n_rows))
    return n_classes, labels, values
