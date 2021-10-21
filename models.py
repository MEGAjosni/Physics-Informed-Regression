import numpy as np

################################################
###### >>>>> The standard SIR model <<<<< ######
################################################
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

#######################################################
###### >>>>> The 7 compartment S3I3R model <<<<< ######
#######################################################
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