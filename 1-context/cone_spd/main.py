# Visualization of the SDP cone for 2x2 matrices
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Constants
    alpha = np.linspace(0, 20, 1000)
    beta = np.linspace(0, 20, 1000)

    # Setting up the constraints
    ALPHA, BETA = np.meshgrid(alpha, beta)
    Gamma_plus = np.sqrt(ALPHA * BETA)
    Gamma_minus = -Gamma_plus

    # Plotting
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(ALPHA, BETA, Gamma_plus, color="gray", shade=False)
    surf = ax.plot_surface(ALPHA, BETA, Gamma_minus, color="gray", shade=False)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel(r"$\gamma$")
    plt.show()
