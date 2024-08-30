"""
    Note: The paper suggests using 50 Gaussians to expand the soft Coulomb potential. 
          However, due to limitations imposed by the `maxfev` parameter in `scipy.optimize.curve_fit`, 
          this code is constrained to expanding the potential with only a few Gaussians. 
          For a more comprehensive basis, please refer to `gaussian_expansion_for_potential_tf.py`.

    This code expands the soft Coulomb potential into 5 Gaussians to facilitate the analytical calculations of the matrix elements. Specifically,

        V(r) = -1 / sqrt(r^2 + a^2) = Σ Ci * exp(-σi * r^2), (0 <= i < 5)

    The parameters Ci and σi are obtained using `scipy.optimize.curve_fit`. 
    
    The results are stored in `V0_expansion_params.npz` with keys "values_C" and "values_σ".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def main():
    # Change to your path
    os.chdir('E:\\XUTTAI\\Desktop\\tdvp\\potential_expansion')

    # Define the target function V(r)
    def V(r):
        a = 1
        return -1 / np.sqrt(r ** 2 + a ** 2)

    # Define the linear combination of n Gaussian functions
    def gaussian_sum(r, *params):
        n = len(params) // 2
        result = np.zeros_like(r)
        for i in range(n):
            a = params[2 * i]
            b = params[2 * i + 1]
            result += - a ** 2 * np.exp(-b * r ** 2)
        return result

    # Generate data
    r = np.linspace(0, 100, 10000)
    V_values = V(r)

    # Initial guess parameters (each Gaussian function has two parameters: C and σ)
    n_gaussians = 5
    initial_guess = [1, 1] * n_gaussians

    # Fit Gaussian functions' parameters
    params_opt, _ = curve_fit(gaussian_sum, r, V_values, p0=initial_guess)

    # Calculate Gaussian expansion values using the fitted parameters
    gaussian_values = gaussian_sum(r, *params_opt)
    gaussian_params = {}
    for i in range(n_gaussians):
        C = - params_opt[2 * i] ** 2
        σ = params_opt[2 * i + 1]
        gaussian_params[f'Gaussian {i+1}'] = {'C': np.complex128(C), 'σ': np.complex128(σ)}

    keys = list(gaussian_params.keys())
    values_C = np.array([gaussian_params[key]['C'] for key in keys], dtype=np.complex128)
    values_σ = np.array([gaussian_params[key]['σ'] for key in keys], dtype=np.complex128)

    # Save the results
    np.savez('V0_expansion_params.npz', keys=keys, values_C=values_C, values_σ=values_σ)

    # Load from .npz file
    V0_expansion_params = np.load('V0_expansion_params.npz')
    loaded_values_C = V0_expansion_params['values_C']
    loaded_values_σ = V0_expansion_params['values_σ']

    print('Loaded C values:', loaded_values_C)
    print('Loaded σ values:', loaded_values_σ)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(r, V_values, label='V(r) = -1 / sqrt(r^2 + a^2)', color='blue', lw=2)
    plt.plot(r, gaussian_values, label=f'Gaussian Sum (n={n_gaussians})', color='red', linestyle='--', lw=2)
    plt.xlabel('r')
    plt.ylabel('Function value')
    plt.title('Comparison between V(r) and Gaussian Sum Expansion')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()


