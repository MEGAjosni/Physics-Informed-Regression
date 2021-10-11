import math
import numpy as np


def derivative_expanded(X, mp, t):

    # *** Description ***
    # Computes the derivative of X using model parameters

    # ************* Input *************
    #
    # t : added vaccinations 
    #
    # X : State vector
    # S : susceptible
    # I1 : regular infected
    # I2 : ICU infected
    # I3 : respirator infected
    # R1 : regular recovered
    # R2 : vaccinated
    # R3 : dead

    # mp : model parameters
    # beta : Infection rate parameter 
    # gamma1 : Rate of recovery for infected
    # gamma2 : Rate of recovery for ICU
    # gamma3 : Rate of recovery for respirator
    # theta : Death rate in ICU
    # phi1 : Rate of hospitalised from infected
    # phi2 : Rate of ICU from hospitalised
    # N : Population
    # ************* Output *************
    #
    # dX : derivative of state vector X. 

    # Extract data
    beta, gamma1, gamma2, gamma3, theta, phi1, phi2 = mp
    N = 5800000
    S, I1, I2, I3, R1, R2, R3 = X

    dX = np.array([
        -((beta * I1) / N + (t / (S + I1 + R1))) * S,  # dS/dt
        (beta * I1 / N) * S - (gamma1 + phi1 + (t / (S + I1 + R1))) * I1,  # dI1/dt
        phi1 * I1 - (gamma2 + phi2) * I2,  # dI2/dt
        phi2 * I2 - (gamma3 + theta) * I3,  # dI3/dt
        gamma1 * I1 + gamma2 * I2 + gamma3 * I3 - (t / (S + I1 + R1)) * R2,  # dR1/dt
        t,  # dR2/dt
        theta * I3,  # dR3/dt
    ])

    return dX


def RK4(
        X_k: np.array,  # Values of the expanded SIR at time t_k
        mp: list,  # Model parameters [beta, gamma, N]
        t: int,  # added vaccinations
        stepsize: int = 1  # t_kp1 - t_k

):
    # *** Description ***
    # Uses Rung Kutta 4 to solve ODE system numerically.

    # *** Output ***
    # X_kp1 [list]:         Values of SIR at time t_kp1

    K_1 = derivative_expanded(X_k, mp, t)
    K_2 = derivative_expanded(X_k + 1 / 2 * stepsize * K_1, mp, t)
    K_3 = derivative_expanded(X_k + 1 / 2 * stepsize * K_2, mp, t)
    K_4 = derivative_expanded(X_k + stepsize * K_3, mp, t)

    X_kp1 = X_k + stepsize / 6 * (K_1 + 2 * (K_2 + K_3) + K_4)

    return X_kp1


def PID_cont(ICU_prev, beta_prev, e_total, e_prev, K):
    I3_hat = 322-150    
    e = ICU_prev-I3_hat
    #print(e)
    e_total = e_total + e
    PID = ((K[0] * e) + (K[1] * e_total) + K[2] * (e - e_prev))
    
    '''
    # Cant handle large exponents:
    if PID > 35:
        new_beta = 0.8 * beta_prev
    elif PID < -35:
        new_beta = 1.2 * beta_prev
    else:
        new_beta = beta_prev * (0.8 + (0.4 / (1 + math.exp(PID))))
        '''
    
    new_beta = beta_prev + PID 
    
    if new_beta < beta_prev * 0.8:
        new_beta = beta_prev * 0.8
    if new_beta > beta_prev * 1.2:
        new_beta = beta_prev * 1.2
    
    if new_beta < 0.06:
        new_beta = 0.06
    if new_beta > 0.25:
        new_beta = 0.25
    #print("PID: ", PID, "Beta: ",new_beta, "Beta_prev: ",beta_prev)
    return new_beta, e, e_total,


def simulateSIR(
        X_0: np.array,  # Initial values of SIR [S_0, I_0, R_0]
        mp: list,  # Model parameters [beta, gamma, N]
        T: np.array,  # Total added vaccinations
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 0.1,  # t_kp1 - t_k
        method=RK4  # Numerical method to be used [function]
):
    # *** Description ***
    # Simulate SIR-model.

    # *** Output ***
    # t [list]:             All points in time simulated
    # SIR [nested list]:    Values of SIR at all points in time t

    SIR = [X_0]

    t = np.arange(start=0, stop=simtime+stepsize/2, step=stepsize)

    for i in range(int(simtime / stepsize)):
        SIR.append(method(SIR[i], mp, T[int(i*stepsize)], stepsize))

    return t, np.array(SIR)


def simulateSIR_PID(
        X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
        mp: list, # Model parameters [beta, gamma, N]
        beta_initial: float, 
        T: list,  # Total added vaccinations
        K: list,  # parameters for penalty function
        simtime: int,  # How many timeunits into the future that should be simulated
        stepsize,  # t_kp1 - t_k
        method=RK4  # Numerical method to be used [function]

):
    SIR = [X_0]
    t = [i * stepsize for i in range(int(simtime / stepsize) + 1)]
    e_total = 0
    e = 0
    error_vals = [0]
    beta_vals = [beta_initial]
    
    for i in range(0, int(simtime / stepsize)):
        new_beta = 0
        
        if i % 70 == 0: # only update beta every 7 days (10*i = one day)
            j = int(i/70)
            
            if i == 0:
                
                new_beta, e, e_total = PID_cont(SIR[0][3],beta_initial, e_total, 0, K)
            else:
                new_beta, e, e_total = PID_cont(SIR[i][3],beta_vals[j], e_total, error_vals[j-1], K)
            
            error_vals.append(e)
            
            
            beta_vals.append(new_beta)
        SIR.append(method(SIR[i], [beta_vals[j-1]]+mp, T[i], stepsize))

    return t, SIR, beta_vals, error_vals


def param_est_expanded_PID(X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
                           mp: list,  # Model parameters [beta, gamma, N]
                           T: list,  # Total added vaccinations
                           simtime: int = 100,  # How many timeunits into the future that should be simulated
                           stepsize: float = 1,  # t_kp1 - t_k
                           method=RK4  # Numerical method to be used [function]

                           ):
    Min_betas = []
    count = 0
    for i in np.linspace(-4, 0, 50):
        for j in np.linspace(-10 / 1000, 0, 50):
            for k in np.linspace(-10 * 80, 0, 50):
                mp[0] = 0.22
                t, State_vec, beta_vals, error_vals = simulateSIR_PID(
                    X_0=X_0,
                    mp=mp,
                    T=T,
                    K=[i, j, k],
                    simtime=simtime,
                    stepsize=1,
                    method=RK4

                )

                count += 1
                if count % 1000 == 0:
                    print("Completed: ", count * 100 / (50 ** 3), "%")
                if max(error_vals) <= 0:
                    Min_betas.append([min(beta_vals), i, j, k])

    opt_parameters = []
    best_beta = 0
    for i in range(len(Min_betas)):
        if Min_betas[i][0] >= best_beta:
            best_beta = Min_betas[i][0]
            opt_parameters = [Min_betas[i][1], Min_betas[i][2], Min_betas[i][3]]
        return opt_parameters, best_beta
