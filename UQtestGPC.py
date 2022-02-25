# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 01:17:47 2022

@author: jonas
"""

import chaospy as cp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tikzplotlib

# Define model
def SIR(x, coords, beta):

    gamma = 1/9;
    
    mp = np.array([beta, gamma])

    # Compute the total population
    N = sum(x) # S + I + R
    
    # Construct system matrix from model    
    A = np.array([
         [-x[1]*x[0]/N,     0],
         [x[1]*x[0]/N,   -x[1]],
         [0,             x[1]]
        ])


    # Compute dxdt.    
    dS, dI, dR = np.array(A) @ np.array(mp).T
    
    return dS, dI, dR

def IC(delta):
    return 1-delta, delta, 0;

# Assume beta to be normal distributed
mu = 0.2;
sigma = 0.01;
dist_beta = cp.Normal(mu, sigma);

# Generate polynomial expansion using Hermite polynomials
P = 8
PCE = cp.generate_expansion(P, dist_beta)

# Compute Gaussian nodes and weights 
Q = 20
G_nodes, weights = cp.generate_quadrature(Q, dist_beta, rule="gaussian")

# Simulate from Gaussian nodes
time_span = np.linspace(0, 200, 1000)
def model_solver(parameters, delta=1e-4):
    return odeint(SIR, IC(delta), time_span, args=(parameters,))

evaluations = np.array([odeint(SIR, IC(delta=0.0001), time_span, args=(G_nodes[0, i],)) for i in range(G_nodes.size)])

# Get statistics (mean and std)
model_approx = cp.fit_quadrature(PCE, G_nodes, weights, evaluations[:, :, 1])
mu = cp.E(model_approx, dist_beta)
std = cp.Std(model_approx, dist_beta)

plt.plot(time_span, mu)
plt.fill_between(time_span, mu-sigma, mu+sigma, alpha=0.3)

plt.xlabel("Time [days]")
plt.ylabel("% of population")
plt.legend(["E(I)", "1 std CI"])
tikzplotlib.save('PCE.tex')