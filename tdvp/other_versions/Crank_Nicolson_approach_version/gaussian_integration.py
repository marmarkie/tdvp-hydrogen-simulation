"""

    This code implements necessary integrations for PTG.

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import os

# B10 without γ
# g_σ(z) = e^(-σ * z^2)
# <g_σ1|z^k * e^(-ν*z^2)|g_σ2>
def integrate_g_σ1_zk_g_σ2(σ1, k, ν, σ2):
    if k % 2 == 1:
        return 0
    else:
        def double_factorial(n):
            result = 1
            for i in range(n, 0, -2):
                result *= i
            return result      
        if (σ1.conjugate() + σ2 + ν) <= 0:
            print('Re(σ* + σ\' + ν) < 0 \n Integration is divergent')
            return 1
        else:
            return double_factorial(k - 1) * np.sqrt(np.pi) * (σ1.conjugate() + σ2 + ν) ** (-(k + 1) / 2) * 2 ** (-k / 2)

# g = z^n * e^(γ−α(x^2+y^2)−βz^2)
class g_basis():
    def __init__(self, n, γ, α, β, direction = 'z'):
        self.n = n
        self.γ = γ
        self.α = α
        self.β = β

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_g_2(g_1, g_2):
    
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    y_integration = x_integration
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n, ν=0, σ2=g_2.β)
    return x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|x^2 + y^2|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
# x,y对称,只用算一半
def integrate_g_1_x2y2_g_2(g_1, g_2):
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=2, ν=0, σ2=g_2.α)
    y_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n, ν=0, σ2=g_2.β)
    return 2 * x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^2|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_z2_g_2(g_1, g_2):
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    y_integration = x_integration
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n + 2, ν=0, σ2=g_2.β)
    return x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|(x^2 + y^2)^2|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_x2y2_2_g_2(g_1, g_2):
    def x4_term():
        x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=4, ν=0, σ2=g_2.α)
        y_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
        z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n, ν=0, σ2=g_2.β)
        return x_integration * y_integration * z_integration
    
    def x2y2_term():
        x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=2, ν=0, σ2=g_2.α)
        y_integration = x_integration
        z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n, ν=0, σ2=g_2.β)
        return x_integration * y_integration * z_integration
    
    return 2 * (x4_term() + x2y2_term()) * np.exp(g_1.γ.conjugate() + g_2.γ)

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|(x^2 + y^2) * z^2|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_x2y2z2_g_2(g_1, g_2):
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=2, ν=0, σ2=g_2.α)
    y_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n + 2, ν=0, σ2=g_2.β)
    return 2 * x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^4|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_z4_g_2(g_1, g_2):
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    y_integration = x_integration
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n + 4, ν=0, σ2=g_2.β)
    return x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^(-2)|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_z_negetive2_g_2(g_1, g_2):
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    y_integration = x_integration
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n - 2, ν=0, σ2=g_2.β)
    return x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^(-2) * (x^2+y^2)|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_z_negetive2_x2y2_g_2(g_1, g_2):
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=2, ν=0, σ2=g_2.α)
    y_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n - 2, ν=0, σ2=g_2.β)
    return 2 * x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ) 

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^2 * (x^2+y^2)|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_x2y2_z2_g_2(g_1, g_2):
    x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=2, ν=0, σ2=g_2.α)
    y_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=0, σ2=g_2.α)
    z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n + 2, ν=0, σ2=g_2.β)
    return 2 * x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ) 

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|H|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_H_g_2(g_1, g_2, V0_params = 0):
    # kinetic term
    def integrate_g_1_T_g_2(g_1, g_2):
        # C5
        # - 1/2 * (d^2/dx^2 + d^2/dy^2) g = (2α − 2α^2(x^2 + y^2)) g
        kinetic_integration = 2 * g_2.α * integrate_g_1_g_2(g_1, g_2) - 2 * g_2.α ** 2 * integrate_g_1_x2y2_g_2(g_1, g_2)
        # C6
        # - 1/2 * (d^2/dz^2) g = (− 1/2 * n(n−1) * z^(-2) + (2n+1)β −2β^2z^2) g
        if g_2.n >= 2:
            kinetic_integration += - g_2.n * (g_2.n - 1) * integrate_g_1_z_negetive2_g_2(g_1, g_2) / 2
        kinetic_integration += (2 * g_2.n + 1) * g_2.β * integrate_g_1_g_2(g_1, g_2) - 2 * g_2.β ** 2 * integrate_g_1_z2_g_2(g_1, g_2)
        return kinetic_integration
    kinetic_integration = integrate_g_1_T_g_2(g_1, g_2) 
    #print('kinetic_integration',kinetic_integration)
    #return kinetic_integration
    # potential term
    # potential expansion V0(r) = Ci * e^(-Ai * (x^2 + y^2 + z^2)) , i from 0 to n_expansion-1
    def integrate_g_1_V0_g_2(V0_params):
        # <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|e^(-Ai * (x^2 + y^2 + z^2))|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
        def integrate_g_1_Ci_Ai_g_2(g_1, g_2, Ai):
            x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=Ai, σ2=g_2.α)
            y_integration = x_integration
            z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n, ν=Ai, σ2=g_2.β)
            return x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)
        V0_integration = 0
        for i in range(V0_params.shape[0]):
            V0_integration += V0_params[i, 0] * integrate_g_1_Ci_Ai_g_2(g_1, g_2, V0_params[i, 1])           
        return V0_integration
    potential_integration = integrate_g_1_V0_g_2(V0_params)
    #print('potential_integration',potential_integration)
    return kinetic_integration + potential_integration

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|(x^2+y^2)H|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_x2y2_H_g_2(g_1, g_2, V0_params):
    # kinetic term
    def integrate_g_1_x2y2_T_g_2(g_1, g_2):
        # - 1/2 * (x^2+y^2) * (d^2/dx^2 + d^2/dy^2) g = (2α(x^2+y^2) − 2α^2(x^2 + y^2)^2) g
        kinetic_integration = 2 * g_2.α * integrate_g_1_x2y2_g_2(g_1, g_2) - 2 * g_2.α ** 2 * integrate_g_1_x2y2_2_g_2(g_1, g_2)
        # - 1/2 * (x^2+y^2) * (d^2/dz^2) g = {− 1/2 * n(n−1) * z^(-2) * (x^2+y^2) + (2n+1)β * (x^2+y^2)−2β^2z^2(x^2+y^2)} g
        if g_2.n >= 2:
            kinetic_integration += - g_2.n * (g_2.n - 1) * integrate_g_1_z_negetive2_x2y2_g_2(g_1, g_2) / 2
        kinetic_integration += (2 * g_2.n + 1) * g_2.β * integrate_g_1_x2y2_g_2(g_1, g_2) - 2 * g_2.β ** 2 * integrate_g_1_x2y2z2_g_2(g_1, g_2)
        return kinetic_integration
    kinetic_integration = integrate_g_1_x2y2_T_g_2(g_1, g_2)
    # potential term
    def integrate_g_1_x2y2_V0_g_2(V0_params):
        # <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|(x^2+y^2) * e^(-Ai * (x^2 + y^2 + z^2))|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
        def integrate_g_1_x2y2_Ci_Ai_g_2(g_1, g_2, Ai):
            x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=2, ν=Ai, σ2=g_2.α)
            y_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=Ai, σ2=g_2.α)
            z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n, ν=Ai, σ2=g_2.β)
            return 2 * x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)
        V0_integration = 0
        for i in range(V0_params.shape[0]):
            V0_integration += V0_params[i, 0] * integrate_g_1_x2y2_Ci_Ai_g_2(g_1, g_2, V0_params[i, 1])           
        return V0_integration
    potential_integration = integrate_g_1_x2y2_V0_g_2(V0_params)
    #print('kinetic_integration',kinetic_integration)
    #print('potential_integration',potential_integration)

    return kinetic_integration + potential_integration

# <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^2H|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
def integrate_g_1_z2_H_g_2(g_1, g_2, V0_params):
    # kinetic term
    def integrate_g_1_z2_T_g_2(g_1, g_2):
        # - 1/2 * z^2 * (d^2/dx^2 + d^2/dy^2) g = (2αz^2 − 2α^2(x^2 + y^2)z^2) g
        kinetic_integration = 2 * g_2.α * integrate_g_1_z2_g_2(g_1, g_2) - 2 * g_2.α ** 2 * integrate_g_1_x2y2_z2_g_2(g_1, g_2)
        # - 1/2 * z^2 * (d^2/dz^2) g = {− 1/2 * n(n−1) + (2n+1)β * z^2 −2β^2z^4} g
        if g_2.n >= 2:
            kinetic_integration += - g_2.n * (g_2.n - 1) * integrate_g_1_g_2(g_1, g_2) / 2
        kinetic_integration += (2 * g_2.n + 1) * g_2.β * integrate_g_1_z2_g_2(g_1, g_2) - 2 * g_2.β ** 2 * integrate_g_1_z4_g_2(g_1, g_2)
        return kinetic_integration
    kinetic_integration = integrate_g_1_z2_T_g_2(g_1, g_2)
    # potential term
    def integrate_g_1_z2_V0_g_2(V0_params):
        # <z^n1 * e^(γ1−α1(x^2+y^2)−β1z^2)|z^2 * e^(-Ai * (x^2 + y^2 + z^2))|z^n2 * e^(γ2−α2(x^2+y^2)−β2z^2)>
        def integrate_g_1_z2_Ci_Ai_g_2(g_1, g_2, Ai):
            x_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=Ai, σ2=g_2.α)
            y_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.α, k=0, ν=Ai, σ2=g_2.α)
            z_integration = integrate_g_σ1_zk_g_σ2(σ1=g_1.β, k=g_1.n + g_2.n + 2, ν=Ai, σ2=g_2.β)
            return x_integration * y_integration * z_integration * np.exp(g_1.γ.conjugate() + g_2.γ)
        V0_integration = 0
        for i in range(V0_params.shape[0]):
            V0_integration += V0_params[i, 0] * integrate_g_1_z2_Ci_Ai_g_2(g_1, g_2, V0_params[i, 1])           
        return V0_integration
    potential_integration = integrate_g_1_z2_V0_g_2(V0_params)
    #print('kinetic_integration',kinetic_integration)
    #print('potential_integration',potential_integration)

    return kinetic_integration + potential_integration