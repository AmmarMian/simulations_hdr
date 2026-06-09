# Hardware resource management for image splitting and sequential processing

from math import sqrt
from typing import Callable, Optional, Tuple, List, Union
from rich.progress import track

from .backend import (
    Array,
    Backend,
    get_data_on_device,
    Unfold2D,
    concatenate,
    make_writable_copy,
    permute,
    dtype_itemsize,
    empty_cache,
)


class ImageRessourceManager:
    """Base class for splitting-based image time series processing.

    Splits an image into a grid of overlapping tiles, processes each tile
    sequentially via process_one_split, then stitches results back together.

    Device transfers, unfolding, and memory cleanup are all driven by
    backend so no subclass overrides are needed for the processing loop.

    Image data must be shaped (n_times, n_channels, height, width).
    Results are returned on the same backend as specified.
    """

    def __init__(
        self,
        image_data: Array,
        window_size: int,
        stride: int,
        process_one_split: Callable,
        backend: str | Backend,
        splitting: Tuple[int, int] = (1, 1),
        verbose: Optional[int] = 1,
    ) -> None:
        assert image_data.ndim == 4, (
            "Data must be 4D: (n_times, n_channels, height, width)"
        )
        self.image_data = image_data
        self.n_times, self.n_channels, self.height, self.width = image_data.shape
        self.window_size = window_size
        self.stride = stride
        self.process_one_split = process_one_split
        self.backend = (
            Backend.from_str(backend) if isinstance(backend, str) else backend
        )
        self.verbose = verbose
        self.n_rows, self.n_cols = splitting
        self._unfold2d = Unfold2D(window_size, stride)
        self.splits_coordinates = compute_splitting_coordinates(
            self.height, self.width, self.n_rows, self.n_cols, self.window_size
        )

    def get_split(self, split_no: int) -> Array:
        assert split_no < self.n_rows * self.n_cols
        row_min, row_max, col_min, col_max = self.splits_coordinates[split_no]
        return get_data_on_device(
            self.image_data[..., row_min:row_max, col_min:col_max],
            self.backend,
        )

    def _unfold(self, split_data: Array) -> Array:
        return self._unfold2d(split_data, self.backend)

    def _delete_temp(self, data) -> None:
        del data
        if self.backend.is_gpu:
            empty_cache(self.backend)

    def _finalize_result(self, result: Array, output_shape) -> Array:
        result = get_data_on_device(result, self.backend)
        return make_writable_copy(self.backend, result.reshape(output_shape))

    def process_all_data(self, *args, **kwargs) -> Array:
        """Process all splits sequentially and merge results."""
        rows_range = range(self.n_rows)
        if self.verbose:
            print("Starting processing")
            rows_range = track(rows_range, description="Processing rows")

        results_to_merge = []
        for row in rows_range:
            result_row = []
            for col in range(self.n_cols):
                split_no = row * self.n_cols + col
                split_data = self.get_split(split_no)
                _, _, split_h, split_w = split_data.shape

                sliding_windows = self._unfold(split_data)
                self._delete_temp(split_data)

                result = self.process_one_split(sliding_windows, *args, **kwargs)
                self._delete_temp(sliding_windows)

                output_h = (split_h - self.window_size) // self.stride + 1
                output_w = (split_w - self.window_size) // self.stride + 1
                if result.ndim == 2:
                    output_shape = (output_h, output_w)
                else:
                    output_shape = (output_h, output_w) + tuple(result.shape[2:])

                result_row.append(self._finalize_result(result, output_shape))
                self._delete_temp(result)

            results_to_merge.append(result_row)

        return concatenate(
            self.backend,
            [concatenate(self.backend, row, axis=1) for row in results_to_merge],
            axis=0,
        )


def _query_vram_mb(backend: Backend) -> float:
    """Return available VRAM in MB for the given GPU backend.

    * **torch-cuda**: uses ``torch.cuda.mem_get_info``.
    * **cupy-cuda**: uses ``cupy.cuda.runtime.memGetInfo``.
    * **jax-cuda / jax-metal**: no public free-memory API; returns a
      conservative 4 096 MB default.  Override via the *vram* parameter.
    """
    if backend.is_torch and backend.is_cuda:
        from torch.cuda import mem_get_info
        return mem_get_info(device=backend.torch_device)[0] / (1024 * 1024)
    if backend.is_cupy:
        try:
            import cupy
            free, _ = cupy.cuda.runtime.memGetInfo()
            return free / (1024 * 1024)
        except ImportError:
            pass
    # jax or any other GPU backend: conservative fallback
    return 4096.0


class ImageGPURessourceManager(ImageRessourceManager):
    """GPU-backed splitting manager.

    Thin wrapper around :class:`ImageRessourceManager` that resolves the
    device, queries available VRAM, and supports automatic splitting.

    Parameters
    ----------
    image_data : Array
    window_size : int
    stride : int
    process_one_split : Callable
    backend : str or Backend
        Any GPU backend (``"torch-cuda"``, ``"cupy"``, ``"jax-cuda"``,
        ``"jax-metal"``).  Defaults to ``"torch-cuda"`` for backward
        compatibility.
    splitting : str or tuple
        ``"auto"`` to compute from available VRAM, or an explicit ``(r, c)``.
    vram : float, optional
        Override the VRAM estimate (MB).  Required for JAX GPU backends which
        have no reliable free-memory query.
    verbose : int
    """

    def __init__(
        self,
        image_data: Array,
        window_size: int,
        stride: int,
        process_one_split: Callable,
        backend: str | Backend = "torch-cuda",
        splitting: Union[str, Tuple[int, int]] = "auto",
        vram: Optional[float] = None,
        verbose: Optional[int] = 1,
    ) -> None:
        resolved = Backend.from_str(backend) if isinstance(backend, str) else backend
        self.dtype = image_data.dtype
        self.vram = vram or _query_vram_mb(resolved)

        if splitting == "auto":
            _, n_channels, height, width = image_data.shape
            splitting = compute_splits_auto(
                height, width, self.dtype, self.vram, window_size, stride, n_channels
            )

        super().__init__(
            image_data,
            window_size,
            stride,
            process_one_split,
            backend=resolved,
            splitting=splitting,
            verbose=verbose,
        )


class ImageCPURessourceManager(ImageRessourceManager):
    """CPU-backed splitting manager.

    Thin wrapper around :class:`ImageRessourceManager` with CPU-appropriate
    defaults: no device transfer, no VRAM management, splitting defaults to
    ``(1, 1)``.  Accepts any CPU-compatible backend (``numpy``, ``torch-cpu``,
    ``jax-cpu``).
    """

    def __init__(
        self,
        image_data: Array,
        window_size: int,
        stride: int,
        process_one_split: Callable,
        backend: str | Backend = "torch-cpu",
        splitting: Tuple[int, int] = (1, 1),
        verbose: Optional[int] = 1,
    ) -> None:
        super().__init__(
            image_data,
            window_size,
            stride,
            process_one_split,
            backend=backend,
            splitting=splitting,
            verbose=verbose,
        )


class OnlineImageResourceManager:
    """Online streaming version for temporal processing.

    Processes each spatial split through all time steps sequentially,
    maintaining state per split.  Reads one time step at a time from a
    numpy memmap so only the active slice is paged into RAM.

    image_data must be a numpy array (or memmap) shaped
    ``(n_times, n_rows, n_cols, n_features)``.
    """

    def __init__(
        self,
        image_data,
        window_size: int,
        stride: int,
        detector,
        backend: str | Backend = "numpy",
        splitting: Tuple[int, int] = (1, 1),
        verbose: Optional[int] = 1,
    ) -> None:
        assert image_data.ndim == 4, (
            "Data must be 4D: (n_times, n_rows, n_cols, n_features)"
        )
        self.memmap = image_data
        self.n_times, self.height, self.width, self.n_channels = image_data.shape
        self.window_size = window_size
        self.stride = stride
        self.detector = detector
        self.backend = Backend.from_str(backend) if isinstance(backend, str) else backend
        self.verbose = verbose
        self.n_rows, self.n_cols = splitting
        self._unfold2d = Unfold2D(window_size, stride)
        self.splits_coordinates = compute_splitting_coordinates(
            self.height, self.width, self.n_rows, self.n_cols, window_size
        )

    def _unfold(self, split_data: Array) -> Array:
        return self._unfold2d(split_data, self.backend)

    def _delete_temp(self, data) -> None:
        del data
        if self.backend.is_gpu:
            empty_cache(self.backend)

    def _finalize_result(self, result: Array, output_shape) -> Array:
        result = get_data_on_device(result, self.backend)
        return make_writable_copy(self.backend, result.reshape(output_shape))

    def _get_slice(
        self,
        t_start: int,
        t_end: int,
        row_min: int,
        row_max: int,
        col_min: int,
        col_max: int,
    ) -> Array:
        """Read a spatial+temporal crop from memmap and return on backend.

        The memmap is always numpy, shaped (t, h, w, features).
        We need (t, features, h, w) on the target backend.
        """
        # .copy() ensures a writable C-contiguous array even when the slice
        # covers the full spatial extent (ascontiguousarray would return a view)
        data_np = self.memmap[t_start:t_end, row_min:row_max, col_min:col_max, :].copy()
        # (t, h, w, features) → (t, features, h, w) using backend-agnostic permute
        data_on_device = get_data_on_device(data_np, self.backend)
        return permute(self.backend, data_on_device, (0, 3, 1, 2))

    def process_all_data(self, *args, **kwargs) -> Array:
        """Process all splits sequentially with temporal streaming."""
        rows_range = range(self.n_rows)
        if self.verbose:
            print("Starting online processing")
            rows_range = track(rows_range, description="Processing rows")

        results_to_merge = []
        for row in rows_range:
            result_row = []
            for col in range(self.n_cols):
                split_no = row * self.n_cols + col
                result = self._process_split_temporal(split_no)
                result_row.append(result)

            results_to_merge.append(result_row)

        return concatenate(
            self.backend,
            [concatenate(self.backend, row, axis=1) for row in results_to_merge],
            axis=0,
        )

    def _process_split_temporal(self, split_no: int) -> Array:
        """Process one spatial split through all time steps.

        Returns
        -------
        Array of shape (output_h, output_w) [+ extra dims if detector outputs them]
        """
        # Get split spatial boundaries
        row_min, row_max, col_min, col_max = self.splits_coordinates[split_no]
        split_h = row_max - row_min
        split_w = col_max - col_min
        output_h = (split_h - self.window_size) // self.stride + 1
        output_w = (split_w - self.window_size) // self.stride + 1

        # Reset detector state for this new sequence
        self.detector.reset_state()

        # Initialize with t=0, 1 — reads only 2 time slices from disk
        split_data_init = self._get_slice(0, 2, row_min, row_max, col_min, col_max)
        sliding_windows_init = self._unfold(split_data_init)  # (n_windows, 2, k², C)
        self._delete_temp(split_data_init)

        res = self.detector.initialize(sliding_windows_init)
        self._delete_temp(sliding_windows_init)

        # Stream through remaining dates one at a time
        for t in range(2, self.n_times):
            split_data_t = self._get_slice(t, t + 1, row_min, row_max, col_min, col_max)
            sliding_windows_t = self._unfold(split_data_t)  # (n_windows, 1, k², C)
            self._delete_temp(split_data_t)

            # Squeeze time dimension: (n_windows, 1, k², C) → (n_windows, k², C)
            sliding_windows_t = sliding_windows_t[:, 0, :, :]
            res = self.detector.update(res, sliding_windows_t)
            self._delete_temp(sliding_windows_t)

        # Reshape result to spatial grid
        if res.ndim == 1:
            output_shape = (output_h, output_w)
        else:
            output_shape = (output_h, output_w) + tuple(res.shape[1:])

        return self._finalize_result(res, output_shape)


class OnlineImageGPURessourceManager(OnlineImageResourceManager):
    """GPU-backed online streaming manager.

    Resolves device, queries VRAM, and supports automatic splitting.
    image_data must be shaped ``(n_times, n_rows, n_cols, n_features)``.

    Parameters
    ----------
    backend : str or Backend
        Any GPU backend.  Defaults to ``"torch-cuda"`` for backward
        compatibility.
    vram : float, optional
        VRAM override in MB.  Required for JAX GPU backends.
    """

    def __init__(
        self,
        image_data,
        window_size: int,
        stride: int,
        detector,
        backend: str | Backend = "torch-cuda",
        splitting: Union[str, Tuple[int, int]] = "auto",
        vram: Optional[float] = None,
        verbose: Optional[int] = 1,
    ) -> None:
        resolved = Backend.from_str(backend) if isinstance(backend, str) else backend
        self.dtype = image_data.dtype
        self.vram = vram or _query_vram_mb(resolved)

        if splitting == "auto":
            _, height, width, n_channels = image_data.shape
            splitting = compute_splits_auto(
                height, width, self.dtype, self.vram, window_size, stride, n_channels
            )

        super().__init__(
            image_data,
            window_size,
            stride,
            detector,
            backend=resolved,
            splitting=splitting,
            verbose=verbose,
        )


def compute_splitting_coordinates(
    height: int,
    width: int,
    n_rows: int,
    n_cols: int,
    windows_size: int,
) -> List[Tuple[int, int, int, int]]:
    """Split an image into overlapping tiles.

    Returns a list of ``(row_min, row_max, col_min, col_max)`` tuples with
    half-window overlap on interior boundaries to avoid edge artefacts.
    """
    splits_list = []
    for i_row in range(n_rows):
        index_row_start = (
            0 if i_row == 0 else int(height / n_rows) * i_row - int(windows_size / 2)
        )
        index_row_end = (
            height
            if i_row == n_rows - 1
            else int(height / n_rows) * (i_row + 1) + int(windows_size / 2)
        )

        for i_col in range(n_cols):
            index_column_start = (
                0 if i_col == 0 else int(width / n_cols) * i_col - int(windows_size / 2)
            )
            index_column_end = (
                width
                if i_col == n_cols - 1
                else int(width / n_cols) * (i_col + 1) + int(windows_size / 2)
            )

            splits_list.append(
                (index_row_start, index_row_end, index_column_start, index_column_end)
            )

    return splits_list


def compute_splits_auto(
    height: int,
    width: int,
    dtype,
    vram: float,
    window_size: int,
    stride: int,
    n_channels: int = 1,
) -> Tuple[int, int]:
    """Compute number of splits given dtype and available VRAM (in MB).

    Uses 20 % of available VRAM to account for simultaneous allocations,
    framework overhead, and other GPU consumers.

    *dtype* accepts both numpy and torch dtypes.
    """
    n_windows = ((height - window_size) // stride + 1) * (
        (width - window_size) // stride + 1
    )
    n_elements = (vram * 0.2 * 1024 * 1024) / dtype_itemsize(dtype)
    n_elements_needed = n_windows * n_channels * window_size * window_size
    n_splits = int(n_elements_needed / n_elements) + 1
    n_rows = int(sqrt(n_splits))
    n_cols = int(n_splits / n_rows) + (1 if n_splits % n_rows else 0)
    return n_rows, n_cols
