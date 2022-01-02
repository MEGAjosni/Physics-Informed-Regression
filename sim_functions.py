import numpy as np
import os

# Set path to dir of file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from models import SIR

def AddNoise(X, mu=0, sigma=1):
    '''

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    from numpy.random import normal

    noise = normal(mu, sigma, X.shape)
    
    return X + noise
    

def SimulateModel(t, x0, mp, model=SIR):
    '''
    Description
    -----------
    Simulates the specified model using the solve_ivp method from scipy.integrate

    Parameters
    ----------
    t : list
        Sample points.
    x0 : list
        Initial value.
    mp : list
        Model parameters.
    model : function, optional
        The model to simulate. The default is SIR.

    Returns
    -------
    np.array
        Simulated state at each sample point.

    '''
    
    # Include support for variable parameters
    MP = []
    for idx, param in enumerate(mp):
        # Check if parameter is already specified as a list and matches number of periods.
        if type(param) == list:
            if len(param) == len(t)-1:
                MP.append(param)
            else:
                print('Error: Dimension of parameter at index', idx, 'does not match the dimension of t;', len(param), 'vs.', len(t)-1)
        else:
            # If parameter is specified as a single value, use that values at each timestep.
            MP.append([param]*(len(t)-1))
    MP = np.array(MP).T
    
    # Run simulation
    from scipy.integrate import solve_ivp
    
    X = np.array([x0])
    for idx, tk in enumerate(t[1:]):
        sol = solve_ivp(fun=model, t_span=(t[idx], t[idx+1]), y0=X[idx, :], t_eval=[tk], args=tuple([MP[idx]])) 
        X = np.concatenate((X, sol.y.T), axis=0)
      
    return X


def LeastSquareModel(t, data, model=SIR, fix_params=None, normalize=False):
    '''
    Description
    -----------
    Estimating model parameters using the method of Ordinary Least Squares.

    Parameters
    ----------
    t : list
        Sample points.
    data : np.array
        Sample states.
    model : function, optional
        The model that should be estimated. The default is SIR.
    fix_params : list, optional
        List with parameters that should be fixed.
        Prototype: [[0, 0.2], [3, 0.1]], fixes parameter 0 with value 0.2 and parameter 3 with value 0.1.

    Returns
    -------
    mp_est : np.array
        Estimated parameters.

    '''
    
    # Convert to numpy and estimate gradient
    data = np.array(data)
    gradient = np.gradient(data, t, axis=0)
    
    # Get number of observations and states
    m, n = data.shape
    
    # Get number of parameters
    k = model(0, data[0], 0, matrix=True).shape[1]

    # Initialize array
    A = np.zeros((m*n, k))

    if normalize:
        # Normalize equations
        std = np.std(gradient, axis=0)
        std[np.where(std == 0)] = 1
        gradient = gradient / std
        for i, state in enumerate(data):
            A[i*n:(i+1)*n, :] = np.array(model(0, state, 0, matrix=True)) * (1/std.reshape((n,1)))
    else:
        for i, state in enumerate(data):
            A[i*n:(i+1)*n, :] = np.array(model(0, state, 0, matrix=True))
    
    # Create right hand side
    y = gradient.flatten()
    
    # Handle fixed parameters
    if fix_params is not None:
        deleted_params = []
        for g in fix_params:
            y -= g[1] * A[:, g[0]]
            deleted_params.append(g[0])

        A = np.delete(A, deleted_params, axis = 1)

    mp_est = np.linalg.lstsq(A, y, rcond=None)[0]
    
    if fix_params is not None:
        for g in sorted(fix_params):
            mp_est = np.insert(mp_est, g[0], g[1])
        
    return mp_est


'''
Probably not gonna need this one...
'''
def NoneNegativeLeastSquares(mp0, t, data, model=SIR, fix_params=None, normalize=False):
    
    
    # Convert to numpy and estimate gradient
    data = np.array(data)
    gradient = np.gradient(data, t, axis=0)
    
    # Get number of observations and states
    m, n = data.shape
    
    # Get number of parameters
    k = model(0, data[0], 0, matrix=True).shape[1]

    # Initialize array
    A = np.zeros((m*n, k))

    if normalize:
        # Normalize equations
        std = np.std(gradient, axis=0)
        std[np.where(std == 0)] = 1
        gradient = gradient / std
        for i, state in enumerate(data):
            A[i*n:(i+1)*n, :] = np.array(model(0, state, 0, matrix=True)) * (1/std.reshape((n,1)))
    else:
        for i, state in enumerate(data):
            A[i*n:(i+1)*n, :] = np.array(model(0, state, 0, matrix=True))
    
    # Create right hand side
    y = gradient.flatten()
        
    # Handle fixed parameters
    if fix_params is not None:
        deleted_params = []
        for g in fix_params:
            y -= g[1] * A[:, g[0]]
            deleted_params.append(g[0])

        A = np.delete(A, deleted_params, axis = 1)
    
    # Newtons algorithm
    
    Q = A.T @ A
    c = -A.T @ y
        
    p = np.linalg.solve(Q, - (Q @ mp0 + c))
    
    mp_est = mp0 + p
    
    if fix_params is not None:
        for g in sorted(fix_params):
            mp_est = np.insert(mp_est, g[0], g[1])
    
    return mp_est
    
    
    
    
    
    
    


