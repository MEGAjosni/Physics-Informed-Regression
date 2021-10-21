import numpy as np
import os

# Set path to dir of file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from models import SIR


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
    
    # Solve ivp
    from scipy.integrate import solve_ivp
    
    sol = solve_ivp(fun=model, t_span=(t[0], t[-1]), y0=x0, t_eval=t, args=tuple([mp])) 
    
    return sol.y.T


def LeastSquareModel(t, data, model=SIR):
    '''
    Description
    -----------
    Estimating model parameters using method of Ordinary Least Squares method.

    Parameters
    ----------
    t : list
        Sample points.
    data : np.array
        Sample states.
    model : function, optional
        The model that should be estimated. The default is SIR.

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

    for i, state in enumerate(data):
        A[i*n:(i+1)*n, :] = model(0, state, 0, matrix=True)
    
    y = gradient.flatten()
    
    mp_est = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return mp_est
