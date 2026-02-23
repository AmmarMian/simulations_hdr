# Ressource management into GPU

from math import sqrt
from typing import Callable, Optional, Union, Tuple, List
import torch
from torch.types import Tensor
from torch.cuda import mem_get_info
from torch.nn import Unfold
from rich.progress import track


class ImageGPURessourceManager(object):
    """Utility class that implements a splitting strategy on an image
    time series to perform tasks sequentially and merge the results.

    Image data assumed to be of shape (height, width, n_features, n_times)
    """

    def __init__(
        self,
        image_data: Tensor,
        window_size: int,
        stride: int,
        process_one_split: Callable,
        device: Optional[torch.device] = torch.device("cuda"),
        splitting: Union[str, Tuple[int, int]] = "auto",
        vram: Optional[float] = None,
        verbose: Optional[int] = 1,
    ) -> None:
        assert image_data.ndim == 4, "Data incompatible"
        self.image_data = image_data
        (
            self.n_times,
            self.n_channels,
            self.height,
            self.width,
        ) = image_data.shape
        self.window_size = window_size
        self.stride = stride
        self.process_one_split = process_one_split
        self.device = device
        self.dtype = image_data.dtype
        self.vram = vram or mem_get_info(device=self.device)[0] / (1024 * 1024)
        self.verbose = verbose
        self.unfold = Unfold(kernel_size=self.window_size, stride=self.stride)

        if splitting == "auto":
            self.n_rows, self.n_cols = compute_splits_auto(
                self.height,
                self.width,
                self.dtype,
                self.vram,
                self.window_size,
                self.stride,
                self.n_channels,
            )
        else:
            assert type(splitting) is tuple, (
                f"Splitting {splitting} should be a tuple of ints"
            )
            self.n_rows, self.n_cols = splitting

        self.splits_coordinates = compute_splitting_coordinates(
            self.height, self.width, self.n_rows, self.n_cols, self.window_size
        )

    def get_split(self, split_no: int) -> Tensor:
        """Get the data to the GPU from the split no"""
        assert split_no < self.n_rows * self.n_cols
        row_min, row_max, col_min, col_max = self.splits_coordinates[split_no]
        # TODO: verify that we shoudl maybe add a +1
        return self.image_data[..., row_min:row_max, col_min:col_max].to(self.device)

    def delete_gpu_data(self, torch_data) -> None:
        del torch_data
        torch.cuda.empty_cache()

    def process_all_data(self, *args, **kwargs) -> Tensor:
        """Process all data sequentially and merge them"""
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

                # Unfolding to get sliding windwos
                n_times, n_channels, split_h, split_w = split_data.shape
                k = self.unfold.kernel_size
                patches = self.unfold(split_data)
                self.delete_gpu_data(split_data)
                self.sliding_windows = (
                    patches.view(n_times, n_channels, k * k, -1)  # (T, C, k², L)
                    .permute(3, 0, 2, 1)  # (L, T, k², C)
                    .contiguous()
                )

                # Processing
                result = self.process_one_split(self.sliding_windows, *args, **kwargs)
                self.delete_gpu_data(self.sliding_windows)

                # Final result reworked as an an image
                # Use the SPLIT dimensions, not the full image dimensions
                output_h = (split_h - self.window_size) // self.stride + 1
                output_w = (split_w - self.window_size) // self.stride + 1
                output_shape = (
                    (output_h, output_w)
                    if result.ndim == 2
                    else (output_h, output_w) + result.shape[2:]
                )

                # Freeing memory
                result_row.append(result.detach().cpu().reshape(output_shape))
                self.delete_gpu_data(result)

            results_to_merge.append(result_row)

        return torch.cat([torch.cat(row, dim=1) for row in results_to_merge], dim=0)


def compute_splitting_coordinates(
    height: int,
    width: int,
    n_rows: int,
    n_cols: int,
    windows_size: int,
) -> List[Tuple[int, int, int, int]]:
    """Split an image into a list of tensors according to windows_size and stride.
    Based on CPU parallel implementation.

    Arguments:
    ----------
    height: Image height
    width: Image width
    n_rows: Number of row splits
    n_cols: Number of column splits
    windows_size: Size of sliding window

    Returns:
    --------
    List of (row_min, row_max, col_min, col_max) tuples
    """
    splits_list = []
    for i_row in range(n_rows):
        if i_row == 0:
            index_row_start = 0
        else:
            index_row_start = int(height / n_rows) * i_row - int(windows_size / 2)
        if i_row == n_rows - 1:
            index_row_end = height
        else:
            index_row_end = int(height / n_rows) * (i_row + 1) + int(windows_size / 2)

        for i_col in range(n_cols):
            if i_col == 0:
                index_column_start = 0
            else:
                index_column_start = int(width / n_cols) * i_col - int(windows_size / 2)
            if i_col == n_cols - 1:
                index_column_end = width
            else:
                index_column_end = int(width / n_cols) * (i_col + 1) + int(
                    windows_size / 2
                )

            splits_list.append(
                (index_row_start, index_row_end, index_column_start, index_column_end)
            )

    return splits_list


def compute_splits_auto(
    height: int,
    width: int,
    dtype: torch.dtype,
    vram: float,
    window_size: int,
    stride: int,
    n_channels: int = 1,
) -> Tuple[int, int]:
    """Compute number of splits (privileging columns when needed) given a dtype and a size of vram available given in MB

    Note: Uses only 20% of available VRAM to account for:
    - Multiple simultaneous allocations (unfold output, processing results, etc.)
    - PyTorch's memory management overhead
    - Other GPU memory consumers
    """
    n_bytes_per_element = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.complex64: 8,
        torch.complex128: 16,
    }
    assert dtype in n_bytes_per_element, f"dtype {dtype} not supported"
    # Number of possible windows in the image given size and stride
    n_windows = ((height - window_size) // stride + 1) * (
        (width - window_size) // stride + 1
    )
    # How many elements can we load in memory (use only 20% of available VRAM for safety)
    # This accounts for multiple allocations happening simultaneously
    n_elements = (vram * 0.2 * 1024 * 1024) / n_bytes_per_element[dtype]
    # Total elements needed = num_windows * channels * window_area
    n_elements_needed = n_windows * n_channels * window_size * window_size
    n_splits = int(n_elements_needed / n_elements) + 1
    n_rows = int(sqrt(n_splits))
    n_cols = int(n_splits / n_rows) + (1 if n_splits % n_rows else 0)
    return n_rows, n_cols
