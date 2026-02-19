# Script to export and show a SAR ITS in pauli representation

import sys
from pathlib import Path
import os

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import argparse
import numpy as np
from utils import plot_pauli

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAR ITS visualization utility.")
    parser.add_argument("path", type=str, help="Path to the numpy stored dataset.")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Whether to export each frame as an image.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        help="Path where to export the images.",
        default="./export/",
    )
    parser.add_argument(
        "--export_type",
        type=str,
        choices=["png", "jpg", "pdf"],
        help="Extension of exported images.",
        default="jpg",
    )
    args = parser.parse_args()

    # Make sure data exists and create export directory as needed
    if not os.path.exists(args.path):
        raise FileNotFoundError("SAR data not found, please verify path!")

    if args.export and not os.path.isdir(args.export_path):
        print(
            f"Export directory {args.export_path} do not exists. Attempting to create it..."
        )
        os.makedirs(args.export_path)
        print("Done.")

    # Laod data and validate format
    its_data = np.load(args.path)
    assert its_data.ndim == 4
    assert its_data.shape[2] == 3
    assert np.iscomplexobj(its_data)

    # Plotting
    for time in range(0, its_data.shape[-1]):
        fig = plt.figure()
        plot_pauli(its_data[..., time])
        if args.export:
            plt.axis("off")
            plt.tight_layout()
            export_path = f"image_{time + 1}.{args.export_type}"
            plt.savefig(
                os.path.join(args.export_path, export_path), bbox_inches="tight"
            )
            print(f"TIme {time} saved to {export_path}.")
        plt.title(f"Image at time {time}")
    plt.show()
