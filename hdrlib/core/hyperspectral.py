# Hyperspectral scenes: download, read, and the pre-processing pipeline.
#
# The scenes are the classic AVIRIS/ROSIS ones distributed by the GIC of the
# Universidad del País Vasco. Each comes as two .mat files — the cube and its
# ground truth — plus, for some, a "corrected" cube with the water-absorption
# bands removed. The corrected version is the one to use: the removed bands
# carry no signal and would only inflate the dimension.
#
# The pipeline is the one every covariance-based segmentation uses:
#
#   image -> remove the global mean -> PCA to n_features bands
#         -> sliding window -> one covariance per pixel
#
# and it is where the dimensional regime of ch:learning becomes concrete: a
# 5x5 window gives 25 samples for 5 to 16 bands, so c = d/N sits between 0.2
# and 0.64. Nowhere near the classical asymptotic regime.

import os
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np

from .backend import (
    Array,
    Backend,
    Unfold2D,
    batched_eigh,
    get_backend_module,
    get_data_on_device,
    to_numpy,
)


__all__ = [
    "SCENES",
    "download_scene",
    "read_scene",
    "remove_global_mean",
    "pca_image",
    "sliding_window_vectorize",
    "unvectorize_labels",
    "covariance_per_pixel",
]


# Each entry: the files to fetch, the key of the cube inside its .mat, the key
# of the ground truth, and the number of classes excluding the undefined one.
SCENES = {
    "indianpines": {
        # Served by the RSLab mirror at Tehran; the historical GIC pages at
        # www.ehu.eus host the same files but are not always reachable.
        "urls": {
            "Indian_pines_corrected.mat":
                "https://rslab.ut.ac.ir/documents/437291/1493606/Indian_pines_corrected.mat/"
                "cb74f75f-a162-9a6a-9c2f-e2980f244a4c?t=1710109885237&download=true",
            "Indian_pines_gt.mat":
                "https://rslab.ut.ac.ir/documents/437291/1493606/Indian_pines_gt.mat/"
                "d4d7756f-c9fe-bb6b-5ca9-759bfab3404a?t=1710110013281&download=true",
            "Indian_pines.mat":
                "https://rslab.ut.ac.ir/documents/437291/1493606/Indian_pines.mat/"
                "865099e1-7483-e1d7-4688-166a42d28575?t=1710110076491&download=true",
        },
        "cube": ("Indian_pines_corrected.mat", "indian_pines_corrected"),
        "cube_raw": ("Indian_pines.mat", "indian_pines"),
        "labels": ("Indian_pines_gt.mat", "indian_pines_gt"),
        "n_classes": 16,
    },
    "salinas": {
        "urls": {
            "Salinas_corrected.mat":
                "https://zenodo.org/records/15771735/files/Salinas_corrected.mat?download=1",
            "Salinas_gt.mat":
                "https://zenodo.org/records/15771735/files/Salinas_gt.mat?download=1",
            "Salinas.mat":
                "https://zenodo.org/records/15771735/files/Salinas.mat?download=1",
        },
        "cube": ("Salinas_corrected.mat", "salinas_corrected"),
        "cube_raw": ("Salinas.mat", "salinas"),
        "labels": ("Salinas_gt.mat", "salinas_gt"),
        "n_classes": 16,
    },
}


def download_scene(name: str, data_path: Union[str, Path], raw: bool = False) -> Path:
    """Fetch a scene's ``.mat`` files if they are not there already.

    Parameters
    ----------
    name : str
        Key of :data:`SCENES`.
    data_path : str or Path
        Destination directory; created if missing.
    raw : bool, optional
        Also fetch the uncorrected cube. Off by default — it is only useful to
        show what the water-absorption bands look like.

    Returns
    -------
    Path
        The directory the files live in.
    """
    if name not in SCENES:
        raise KeyError(f"Unknown scene {name!r}; known: {sorted(SCENES)}")
    scene = SCENES[name]
    data_path = Path(data_path).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    wanted = {scene["cube"][0], scene["labels"][0]}
    if raw:
        wanted.add(scene["cube_raw"][0])

    for filename, url in scene["urls"].items():
        if filename not in wanted:
            continue
        destination = data_path / filename
        if destination.exists():
            continue
        print(f"Downloading {filename} -> {destination}")
        urlretrieve(url, destination)
        # Some mirrors sit behind a challenge page that answers 200 with HTML.
        # Left alone, the result is a file named .mat that scipy rejects much
        # later with an unhelpful message, so check it here.
        with open(destination, "rb") as handle:
            head = handle.read(4)
        if not head.startswith(b"MATL") and head[:2] != b"\x50\x4b":
            destination.unlink()
            raise RuntimeError(
                f"{url}\ndid not return a MATLAB file — the mirror probably "
                f"served a challenge or error page. Download {filename} by "
                f"hand into {data_path} and run again."
            )
    return data_path


def read_scene(
    name: str, data_path: Union[str, Path], corrected: bool = True
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Read a scene as ``(cube, labels, n_classes)``.

    The cube is ``(n_rows, n_columns, n_bands)`` in float64 and the labels are
    ``(n_rows, n_columns)`` integers, with ``0`` marking the undefined zones.
    Those zones are kept in the cube — they still contribute samples to their
    neighbours' windows — but are excluded from every score, since their
    ground truth is not reliable.
    """
    from scipy.io import loadmat  # local: scipy is only needed to read .mat

    scene = SCENES[name]
    data_path = Path(data_path).expanduser()
    cube_file, cube_key = scene["cube"] if corrected else scene["cube_raw"]
    labels_file, labels_key = scene["labels"]

    cube = loadmat(data_path / cube_file)[cube_key].astype(np.float64)
    labels = loadmat(data_path / labels_file)[labels_key].astype(np.int64)
    return cube, labels, scene["n_classes"]


def remove_global_mean(cube: Array, backend: Union[str, Backend] = "numpy") -> Array:
    """Subtract the mean spectrum of the whole image.

    Done before the PCA and before the windowing: the covariances that follow
    are covariances of the *deviation* from the scene's average spectrum, which
    is what makes them comparable from one pixel to another.
    """
    be = get_backend_module(backend)
    return cube - be.mean(cube, axis=(0, 1), keepdims=True)


def pca_image(
    cube: Array, n_features: int, backend: Union[str, Backend] = "numpy"
) -> Array:
    """Keep the ``n_features`` leading principal components of a cube.

    A handful of principal directions represent these scenes well, and the
    reduction is what brings the dimension into a range where a small window
    can carry a covariance at all: 200 bands against 25 samples is hopeless,
    5 bands against 25 is merely difficult.

    Returns ``(n_rows, n_columns, n_features)``.
    """
    be = get_backend_module(backend)
    n_rows, n_columns, n_bands = cube.shape
    if n_features >= n_bands:
        return cube

    flat = be.reshape(cube, (n_rows * n_columns, n_bands))
    flat = flat - be.mean(flat, axis=0, keepdims=True)
    covariance = be.swapaxes(flat, -1, -2) @ flat / flat.shape[0]
    _, eigenvectors = batched_eigh(backend, covariance)
    # eigh returns ascending eigenvalues: the leading components are last.
    leading = eigenvectors[:, -n_features:]
    return be.reshape(flat @ leading, (n_rows, n_columns, n_features))


def sliding_window_vectorize(
    cube: Array,
    window_size: int,
    stride: int = 1,
    backend: Union[str, Backend] = "numpy",
) -> Array:
    """One block of samples per pixel, from its square neighbourhood.

    Wraps :class:`hdrlib.core.backend.Unfold2D`, which is GPU-native on torch
    (``torch.nn.Unfold``) and falls back to ``sliding_window_view`` elsewhere.
    This is the only step of the pipeline that touches the whole image at once,
    so it is the only one where the backend can plausibly matter.

    Parameters
    ----------
    cube : Array of shape (n_rows, n_columns, n_features)
    window_size : int
        Odd, so that the window is centred on its pixel.
    stride : int, optional
        Step between two windows. One keeps every pixel.

    Returns
    -------
    Array of shape (n_pixels, window_size**2, n_features)
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd so the window has a centre")
    be = get_backend_module(backend)
    # Unfold2D wants (n_times, n_channels, height, width); one "time", one
    # channel per feature.
    moved = be.swapaxes(be.swapaxes(cube, -1, -2), -2, -3)[None, ...]
    windows = Unfold2D(window_size, stride)(moved, backend)
    # (n_pixels, 1, window**2, n_features) -> drop the time axis
    return windows[:, 0]


def unvectorize_labels(
    labels: np.ndarray, n_rows: int, n_columns: int, window_size: int, stride: int = 1
) -> np.ndarray:
    """Fold per-window predictions back onto the image grid.

    The windowed image is smaller than the original by ``window_size - 1`` in
    each direction: a window needs to fit entirely inside the scene, so the
    border pixels have no prediction.
    """
    new_rows = -(-(n_rows - window_size + 1) // stride)
    new_columns = -(-(n_columns - window_size + 1) // stride)
    return labels.reshape((new_rows, new_columns))


def crop_labels(
    labels: np.ndarray, window_size: int, stride: int = 1
) -> np.ndarray:
    """Crop a ground truth to the pixels a sliding window actually predicts."""
    half = window_size // 2
    cropped = labels[half : labels.shape[0] - half, half : labels.shape[1] - half]
    if stride > 1:
        cropped = cropped[::stride, ::stride]
    return cropped


def covariance_per_pixel(
    windows: Array, backend: Union[str, Backend] = "numpy"
) -> Array:
    """Sample covariance of every window, centred on the window's own mean."""
    be = get_backend_module(backend)
    centred = windows - be.mean(windows, axis=-2, keepdims=True)
    return be.swapaxes(centred, -1, -2) @ centred / windows.shape[-2]
