import numpy as np

# %%
'''
################################################
###### >>>>> The standard SIR model <<<<< ######
################################################
'''
def SIR(t, x, mp, matrix=False):
    '''
    Parameters
    ----------
    t : list
        Sample times - Only specified by ode solver.
        Prototype: t = [t1, t2, ..., tn]
    x : list
        State of the model.
        Prototype: x = [S, I, R].
    mp: list
        Parameters of the model.
        Prototype: mp = [beta, gamma].
    matrix : Boolean, optional
        If true, returns system matrix. The default is False.

    Returns
    -------
    np.array
        Default, dxdt.

    '''

    # Compute the total population
    N = sum(x) # S + I + R
    
    # Construct system matrix from model    
    A = np.array([
         [-x[1]*x[0]/N,     0],
         [x[1]*x[0]/N,   -x[1]],
         [0,             x[1]]
        ])

    # Return A if desired
    if matrix:
        return A
    
    # Compute dxdt.    
    dXdt = np.array(A) @ np.array(mp).T
    
    return dXdt

# %%
'''
###############################################
###### >>>>> Dark number SIR model <<<<< ######
###############################################
'''
def DarkNumberSIR(t, x, mp, matrix=False):
    '''
    Parameters
    ----------
    t : list
        Sample times - Only specified by ode solver.
        Prototype: t = [t1, t2, ..., tn]
    x : list
        State of the model.
        Prototype: x = [S, I1, I2, R].
    mp: list
        Parameters of the model.
        Prototype: mp = [beta, gamma].
    matrix : Boolean, optional
        If true, returns system matrix. The default is False.

    Returns
    -------
    np.array
        Default, dxdt.

    '''

    # Compute the total population
    N = sum(x) # S + I + R
    
    # Construct system matrix from model    
    A = np.array([
         [-(x[1]+x[2])*x[0]/N,     0],
         [x[1]*x[0]/N,   -x[1]],
         [x[2]*x[0]/N,   -x[2]],
         [0,              x[1]]
        ])

    # Return A if desired
    if matrix:
        return A
    
    # Compute dxdt.    
    dXdt = np.array(A) @ np.array(mp).T
    
    return dXdt


# %%
'''
#######################################################
###### >>>>> The 7 compartment S3I3R model <<<<< ######
#######################################################
'''
def S3I3R(t, x, mp, matrix=False):
    '''
    Parameters
    ----------
    t : list
        Sample times - Only specified by ode solver.
        Prototype: t = [t1, t2, ..., tn]
    x : list
        State of the model.
        Prototype: x = [S, I1, I2, I3, R1, R2, R3].
    mp: list
        Parameters of the model.
        Prototype: mp = [beta, gamma1, gamma2, gamma3, phi1, phi2, theta, tau].
    matrix : Boolean, optional
        If true, returns system matrix. The default is False.

    Returns
    -------
    np.array
        Default, dxdt.

    '''
    
    # Compute the total population
    N = sum(x) # S + I1 + I2 + I3 + R1 + R2 + R3

    # Construct system matrix from model
    A = np.array([
         [-x[0]*x[1]/N, 0, 0, 0, 0, 0, 0, -x[0]/(x[0]+x[1]+x[4])],
         [x[1]*x[0]/N,  -x[1], 0, 0, -x[1], 0, 0, -x[1]/(x[0]+x[1]+x[4])],
         [0, 0, -x[2], 0, x[1], -x[2], 0, 0],
         [0, 0, 0, -x[3], 0, x[2], -x[3], 0],
         [0, x[1], x[2], x[3], 0, 0, 0, -x[4]/(x[0]+x[1]+x[4])],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, x[3], 0]
        ])

    # Return A if desired
    if matrix:
        return A

    # Compute dxdt
    dxdt = np.array(A) @ np.array(mp).T
    
    return dxdt


# %%
'''
#################################################
###### >>>>> Triple Region SIR model <<<<< ######
#################################################
'''
def TripleRegionSIR(t, x, mp, matrix=False):
    '''
    Parameters
    ----------
    t : list
        Sample times - Only specified by ode solver.
        Prototype: t = [t1, t2, ..., tn]
    x : list
        State of the model.
        Prototype: x = [S1, I1, R1, S2, I2, R2, S3, I3, R3].
    mp: list
        Parameters of the model.
        Prototype: mp = [beta11, beta12, beta13, beta21, beta22, beta23, beta31, beta32, beta33, gamma].
    matrix : Boolean, optional
        If true, returns system matrix. The default is False.

    Returns
    -------
    np.array
        Default, dxdt.

    '''
    
    # Compute the total population of each region
    N = [sum(x[3*i:3*i+3]) for i in range(3)]

    # Construct system matrix from model
    A = np.array([
         [-x[0]*x[1]/N[0], 0, 0, -x[3]*x[4]/N[1], 0, 0, -x[6]*x[7]/N[2], 0, 0,  0],
         [ x[0]*x[1]/N[0], 0, 0,  x[3]*x[4]/N[1], 0, 0,  x[6]*x[7]/N[2], 0, 0, -x[1]],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, x[1]],
         [0, -x[0]*x[1]/N[0], 0, 0, -x[3]*x[4]/N[1], 0, 0, -x[6]*x[7]/N[2], 0,  0],
         [0,  x[0]*x[1]/N[0], 0, 0,  x[3]*x[4]/N[1], 0, 0,  x[6]*x[7]/N[2], 0, -x[4]],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, x[4]],
         [0, 0, -x[0]*x[1]/N[0], 0, 0, -x[3]*x[4]/N[1], 0, 0, -x[6]*x[7]/N[2],  0],
         [0, 0,  x[0]*x[1]/N[0], 0, 0,  x[3]*x[4]/N[1], 0, 0,  x[6]*x[7]/N[2], -x[7]],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, x[7]],
        ])

    # Return A if desired
    if matrix:
        return A

    # Compute dxdt
    dxdt = np.array(A) @ np.array(mp).T
    
    return dxdt


# %%
'''
################################################
###### >>>>> Multivariant SIR model <<<<< ######
################################################
'''
def MultivariantSIR(t, x, mp, matrix=False):
    '''
    Parameters
    ----------
    t : list
        Sample times - Only specified by ode solver.
        Prototype: t = [t1, t2, ..., tn]
    x : list
        State of the model.
        Prototype: x = [S, I1, I2, ..., In, R].
    mp: list
        Parameters of the model.
        Prototype: mp = [beta1, beta2, ..., betan, gamma1, gamma2, ..., gamman].
    matrix : Boolean, optional
        If true, returns system matrix. The default is False.

    Returns
    -------
    np.array
        Default, dxdt.

    '''
    
    # Compute the total population
    N = sum(x) # S + I1 + I2 + ... + In + R
    
    # Construct system matrix from model
    
    n = len(x)
    m = int(n-2)
    
    A = np.zeros((n, 2*m))
    
    for i in range(m):
        A[0, i] = -x[0]*x[i+1]/N
        A[n-1, m+i] = x[i+1]
        A[i+1, i], A[i+1, i+2] = x[0]*x[i+1]/N, -x[i+1]
    
    # Return A if desired
    if matrix:
        return A

    # Compute dxdt
    dxdt = np.array(A) @ np.array(mp).T
    
    return dxdt
