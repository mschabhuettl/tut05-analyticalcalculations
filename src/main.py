import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_e, hbar, pi

# Function to calculate the normalization constant
def normalization_constant(L):
    return np.sqrt(2 / L)

# Function to calculate the wave function Psi_n for particle in a box
def psi_n(x, n, L):
    N = normalization_constant(L)
    return N * np.sin(n * pi * x / L)

# Function to calculate the probability density |Psi_n|^2
def psi_n_squared(x, n, L):
    return np.abs(psi_n(x, n, L))**2

# Function to calculate the energy levels E_n
def energy_n(n, m, L):
    return (hbar**2 * pi**2 * n**2) / (2 * m * L**2)

# Parameters
L = 4.0e-10  # Length of the box in meters (4.0 Å)
m = m_e  # Mass of the electron
n_values = [1, 2, 3, 4]  # Quantum numbers

# Create x values
x = np.linspace(0, L, 1000)

# Plotting
fig, axes = plt.subplots(len(n_values), 2, figsize=(14, 10))
fig.suptitle("Teilchen im Kasten: Wellenfunktionen und Wahrscheinlichkeitsdichten", fontsize=16)

for i, n in enumerate(n_values):
    psi = psi_n(x, n, L)
    psi_sq = psi_n_squared(x, n, L)
    E = energy_n(n, m, L)

    # Plot Psi_n
    axes[i, 0].plot(x, psi)
    axes[i, 0].set_title(f"$\\Psi_{n}^{{\\mathrm{{TIK}}}}(x)$ für n={n}")
    axes[i, 0].set_xlabel("Position x (m)")
    axes[i, 0].set_ylabel(f"$\\Psi_{n}(x)$")

    # Plot |Psi_n|^2
    axes[i, 1].plot(x, psi_sq)
    axes[i, 1].set_title(f"$|\\Psi_{n}^{{\\mathrm{{TIK}}}}(x)|^2$ für n={n}")
    axes[i, 1].set_xlabel("Position x (m)")
    axes[i, 1].set_ylabel(f"$|\\Psi_{n}(x)|^2$")

    # Display energy level
    energy_text = f"Energie $E_{n}$: {E:.2e} J"
    axes[i, 1].annotate(energy_text, xy=(0.6, 0.8), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
