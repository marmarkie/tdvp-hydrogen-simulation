"""
    This code uses the finite difference method to solve the ground state radial wave function R(r) 
    for a hydrogen atom under a 3D soft Coulomb potential. The potential is defined as V(r) = -1 / np.sqrt(r^2 + a^2).

    Let u(r) = r * R(r). The radial Schrödinger equation is given by:

        -1/2 * (d^2/dr^2)u(r) + V(r) * u(r) = E * u(r)

    The result is stored in `soft_coulomb_potential_radial_solution.npz`.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os

def main():
    # Change to your path
    os.chdir('E:\\XUTTAI\\Desktop\\tdvp\\groundstate_generator')

    # Parameters
    a = 1
    N = 2000
    r_max = 20  # r_max in units of Bohr radius (a0)
    r = np.linspace(0.01, r_max, N)  # Avoid singularity at r=0
    dr = r[1] - r[0]

    # Potential term V(r), in atomic units is -1/r
    V = -1 / np.sqrt(r**2 + a**2)

    # Construct the Hamiltonian matrix
    diagonal = -1.0 / dr**2 + V
    off_diagonal = 0.5 * np.ones(N - 1) / dr**2

    # Diagonalize the Hamiltonian matrix
    H = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = la.eigh(H)

    # Ground state wave function
    u_r = abs(eigenvectors[:, 0])
    psi = u_r / r
    psi /= max(psi)    

    # Save the results
    np.savez('soft_coulomb_potential_radial_solution.npz', r=r, y=psi)

    # Plot the ground state wave function
    plt.plot(r, psi)
    plt.title('Ground State Radial Wave Function $\\psi(r)$')
    plt.xlabel('Radius $r$ (in units of $a_0$)')
    plt.ylabel('Wave Function $\\psi(r)$')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
