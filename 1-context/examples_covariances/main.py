# Illustation of different regime of covariance matrices


from typing import Optional
from matplotlib.figure import Figure
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from matplot2tikz import clean_figure, save
import os


def save_tikz_with_path(filepath: str, fig: Optional[Figure] = None):
    if not os.path.exists(os.path.dirname(filepath)):
        raise FileNotFoundError(
            f"Directory {os.path.dirname(filepath)} does not exist!"
        )
    if fig is not None:
        clean_figure(fig)
    save(filepath)
    print(f"Tikz code saved to {filepath}")


def numpy_to_pgf_matrix_string(data, format_spec=".6f"):
    """
    Convert a 2D numpy array to pgfplots matrix format string.

    Parameters:
    -----------
    data : numpy.ndarray
        2D numpy array to convert
    format_spec : str, optional
        Number format specification (default: ".6f")

    Returns:
    --------
    str : Matrix data formatted for pgfplots matrix plot (x y z format)
    """
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")

    rows, cols = data.shape
    matrix_str = ""

    # Write data in x y z format for matrix plot
    for i in range(rows):
        for j in range(cols):
            matrix_str += f"{j} {i} {data[i, j]:{format_spec}}\n"

    return matrix_str.rstrip()  # Remove trailing newline


def write_pgfplots_matrix(
    data,
    tex_filename,
    title="",
    xlabel="",
    ylabel="",
    colormap="viridis",
    width="8cm",
    height="6cm",
    colorbar=True,
    grid=False,
    format_spec=".6f",
    data_filename=None,
    verbose=True,
):
    """
    Generate pgfplots LaTeX code for a matrix plot with embedded data.

    Parameters:
    -----------
    data : numpy.ndarray
        2D numpy array to plot
    tex_filename : str
        Output LaTeX filename
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    colormap : str, optional
        PGFPlots colormap name (default: "viridis")
        Options: viridis, plasma, hot, cool, jet, etc.
    width : str, optional
        Plot width (default: "8cm")
    height : str, optional
        Plot height (default: "6cm")
    colorbar : bool, optional
        Whether to show colorbar (default: True)
    grid : bool, optional
        Whether to show grid (default: False)
    format_spec : str, optional
        Number format specification (default: ".6f")
    data_filename : str, optional
        Internal data file name (default: None, generates unique name based on tex_filename)
    verbose : bool, optional
        Whether to print logging information (default: True)

    Returns:
    --------
    str : The filename of the created LaTeX file
    """
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Ensure .tex extension
    if not tex_filename.endswith(".tex"):
        tex_filename += ".tex"

    # Generate unique data filename if not provided
    if data_filename is None:
        import os

        base_name = os.path.splitext(os.path.basename(tex_filename))[0]
        data_filename = f"{base_name}_matrix.dat"

    rows, cols = data.shape

    if verbose:
        print(f"Processing matrix data: {rows}x{cols}")
        print(
            f"Data range: [{np.min(data):{format_spec}}, {np.max(data):{format_spec}}]"
        )

    # Get matrix data string
    matrix_data = numpy_to_pgf_matrix_string(data, format_spec)

    # Build axis options
    axis_options = [
        f"width={width}",
        f"height={height}",
        "enlargelimits=false",
        "axis equal image",
        f"colormap/{colormap}",
        f"xmin=-0.5, xmax={cols - 0.5}",
        f"ymin=-0.5, ymax={rows - 0.5}",
    ]

    if colorbar:
        axis_options.append("colorbar")

    if grid:
        axis_options.append("grid=major")

    if title:
        axis_options.append(f"title={{{title}}}")

    if xlabel:
        axis_options.append(f"xlabel={{{xlabel}}}")

    if ylabel:
        axis_options.append(f"ylabel={{{ylabel}}}")

    # Write LaTeX code
    if verbose:
        print(f"Writing LaTeX code to: {tex_filename}")

    with open(tex_filename, "w") as f:
        f.write("% PGFPlots matrix plot generated from numpy array\n")
        f.write(f"% Matrix size: {rows}x{cols}\n")
        f.write(
            f"% Data range: [{np.min(data):{format_spec}}, {np.max(data):{format_spec}}]\n\n"
        )

        # Start with tikzpicture only
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")

        # Write axis options
        for i, option in enumerate(axis_options):
            if i == len(axis_options) - 1:
                f.write(f"    {option}\n")
            else:
                f.write(f"    {option},\n")

        f.write("]\n\n")

        # Write matrix plot command with inline data
        f.write("\\addplot[\n")
        f.write("    matrix plot,\n")
        f.write(f"    mesh/cols={cols},\n")
        f.write("    point meta=explicit\n")
        f.write("] table[meta=z] {\n")
        f.write("x y z\n")
        f.write(matrix_data)
        f.write("\n};\n\n")

        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n")

    if verbose:
        print(f"Successfully saved LaTeX code to: {tex_filename}")
        print(f"Matrix plot uses colormap: {colormap}")
        print(f"Data embedded inline (no external file)")

    return tex_filename


def numpy_to_matrix_plot(data, base_filename, **kwargs):
    """
    Convenience function to export numpy array as pgfplots matrix plot.

    Parameters:
    -----------
    data : numpy.ndarray
        2D numpy array to export
    base_filename : str
        Base filename (extension will be added automatically)
    **kwargs : dict
        Additional arguments passed to write_pgfplots_matrix()

    Returns:
    --------
    str : The filename of the created LaTeX file
    """
    verbose = kwargs.get("verbose", True)
    if verbose:
        print(f"Starting numpy to matrix plot conversion...")
        print(f"Target filename: {base_filename}")

    result = write_pgfplots_matrix(data, base_filename, **kwargs)

    if verbose:
        print(f"Matrix plot export completed successfully!")

    return result


if __name__ == "__main__":
    # Constant(s)
    d = 7
    rho = 0.8

    # Case 1: identity matrix
    fig = plt.figure()
    cov_id = np.eye(d)
    write_pgfplots_matrix(
        cov_id, "exports/identity.tex", xlabel=r"variable $i$", ylabel=r"variable $j$"
    )
    plt.imshow(cov_id, aspect="auto")
    plt.xlabel(r"variable $i$")
    plt.ylabel(r"variable $j$")
    plt.colorbar()

    # Case 2 : Toeplitz matrix
    cov_toeplitz = toeplitz(np.power(rho, np.arange(0, d)))
    write_pgfplots_matrix(
        cov_toeplitz, "exports/toeplitz", xlabel=r"variable $i$", ylabel=r"variable $j$"
    )
    fig = plt.figure()
    plt.matshow(cov_toeplitz, aspect="auto", fignum=0)
    plt.xlabel(r"variable $i$")
    plt.ylabel(r"variable $j$")
    plt.colorbar()

    # Case 3: random
    cov_rd = np.empty((d, d))
    cov_rd[np.tril_indices(d)] = 2 * np.random.rand(int(d * (d + 1) / 2)) - 1
    cov_rd[np.triu_indices(d)] = cov_rd[np.tril_indices(d)]
    cov_rd[np.diag_indices(d)] = 1
    cov_rd = 0.5 * (cov_rd + cov_rd.T)
    write_pgfplots_matrix(
        cov_rd, "exports/random", xlabel=r"variable $i$", ylabel=r"variable $j$"
    )
    fig = plt.figure()
    plt.matshow(cov_rd, aspect="auto", fignum=0)
    plt.xlabel(r"variable $i$")
    plt.ylabel(r"variable $j$")
    plt.colorbar()

    # Show figures
    plt.show()
