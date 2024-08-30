# Updating the README content to include the mathematical formulas using Markdown syntax.

# TDVP Hydrogen Simulation

This repository implements a TDVP (Time-Dependent Variational Principle) simulation based on the paper ["Simulation of a Hydrogen Atom in a Laser Field Using the Time-Dependent Variational Principle"](https://link.aps.org/doi/10.1103/PhysRevE.101.023313).

# Requirements

```python
numpy==1.20.0
matplotlib==3.7.4
sympy==1.13.1
tensorflow==2.5.0
scipy==1.10.1
```

## Overview

### Potential Expansion
- **`gaussian_expansion_for_potential.py`**: Expands the soft Coulomb potential into Gaussian bases using `scipy.optimize.curve_fit`. The potential is expressed as:
  
  $$V(r) = -\\frac{1}{\\sqrt{r^2 + a^2}} = \\sum_{i=0}^{n_{\\text{V}}} C_i \\cdot \\exp(-\\sigma_i \\cdot r^2)$$
  
- **`gaussian_expansion_for_potential_tf.py`**: Utilizes TensorFlow's optimizer for potential expansion.

### Ground State Generation
- **`groundstate_solver.py`**: Uses the finite difference method to solve the ground state radial wave function \( R(r) \) for a hydrogen atom under a 3D soft Coulomb potential.
- **`gaussian_expansion_for_groundstate.py`**: Expands the ground state into Gaussian bases using `scipy.optimize.curve_fit`. The ground state is represented as:
  
$$
\psi_{\text{groundstate}}(r) = \sum_{i=0}^{n_{\text{gaussians}}} g_i
$$

  
- **`gaussian_expansion_for_groundstate_tf.py`**: Uses TensorFlow's optimizer for ground state expansion.

### TDVP Implementation
- **`gaussian_integration.py`**: Implements the necessary integrations for `tdvp.py`.
- **`tdvp.py`**: Implements the PTG-basis to simulate a hydrogen atom under a soft Coulomb potential, starting from the ground state.

## Other Versions
- **`tdvp_sympy_version.py`**: Aims to achieve the same functionality as `tdvp.py`, utilizing SymPy to parse the kernel function and operator, and to calculate Gaussian integrations through recursion.
- **`tdvp_crank_nicolson.py`**: Adds the Crank-Nicolson approach to update the coefficients \( \\gamma \).

For more detailed information, please refer to the source code files.
"""

