import numpy as np
import torch
from scipy.integrate import odeint

def SIR(X, t, beta, gamma):

    S, I, R = X

    dSdt = - beta*S*I
    dIdt = beta*S*I - gamma*I
    dRdt = gamma*I

    return [dSdt, dIdt, dRdt]


def S3I3R(X, t, beta, gamma1, gamma2, gamma3, tau, theta, phi1, phi2):
    
    S, I1, I2, I3, R1, R2, R3 = X

    dSdt = - beta*S*I1 - tau*S/(S+I1+R1)
    dI1dt = beta*S*I1 - (gamma1 + tau/(S+I1+R1) + phi1)*I1
    dI2dt = phi1*I1 - (gamma2 + phi2)*I2
    dI3dt = phi2*I2 - (gamma3 + theta)*I3
    dR1dt = gamma1*I1 + gamma2*I2 + gamma3*I3 - tau*R1/(S+I1+R1)
    dR2dt = tau
    dR3dt = theta*I3

    return [dSdt, dI1dt, dI2dt, dI3dt, dR1dt, dR2dt, dR3dt]


class ODESimulator:
    def __init__(self, f, params, X0, time_points, noise_level=None, shuffle=False):
        """
        Generate data from ODE model defined by f.
        """
        
        # Add 0.0 to start of time_points
        time_points = [0.0] + list(time_points) if time_points is not None else None
        self.td = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1)[1:] if time_points is not None else None
        self.Xd_clean = torch.tensor(odeint(f, X0.squeeze(), time_points, params), dtype=torch.float32)[1:] if time_points is not None else None
       
        # Add noise
        if noise_level:
            self.Xd = self.Xd_clean.clone() + noise_level * torch.mean(abs(self.Xd_clean)) * torch.randn(*self.Xd_clean.shape)
        else:
            self.Xd = self.Xd_clean.clone()

        # Shuffle data
        if shuffle:
            idx = torch.randperm(self.Xd.shape[0])
            self.td, self.Xd, self.Xd_clean = self.td[idx], self.Xd[idx], self.Xd_clean[idx]

    def __len__(self):
        return len(self.Xd)
    
    def __getitem__(self, idx):
        return self.td[idx], self.Xd[idx]