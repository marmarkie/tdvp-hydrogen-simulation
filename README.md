# tdvp hydrogen simulation
This repo tries to implement tdvp simulation based on the paper ["Simulation of a hydrogen atom in a laser ﬁeld using the time-dependent variational principle"](https://link.aps.org/doi/10.1103/PhysRevE.101.023313).
In the potential expansion folder, gaussian_expansion_for_potential.py expands soft Soft Coulomb potential into gaussain bases via scipy.optimize.curve_fit
that is
  V(r) = -1 / sqrt(r^2 + a^2) = Σ Ci * exp(-σi * r^2), (0 <= i < 5)
gaussian_expansion_for_potential_tf.py uses optimizer in tensorflow for potential expansion.

In the groundstate_generator folder, groundstate_solver.py uses the finite difference method to solve the ground state radial wave function R(r) for a hydrogen atom under a 3D soft Coulomb potential.
gaussian_expansion_for_groundstate.py expands groundstate into gaussain bases via scipy.optimize.curve_fit
that is
  ψ_groundstate(r) = Σ gi , (0 <= i < n_gaussians)
gaussian_expansion_for_groundstate_tf.py uses optimizer in tensorflow for groundstate expansion.

gaussian_integration.py implements necessary integrations for tdvp.py.
tdvp.py implements the PTG-basis to simulate a hydrogen atom under a soft Coulomb potential, starting from the groundstate.


In the other_versions folder, tdvp_sympy_version.py and tdvp_crank_nicolson.py 旨在实现与tdvp.py相同的功能. 
tdvp_sympy_version.py utilizes SymPy to parse the kernel function and operator, 
and to calculate Gaussian integrations through recursion.
tdvp_crank_nicolson.py adds Crank-Nicolson approach to update the coefﬁcients γ.

More detailed information are provided in source codes.
