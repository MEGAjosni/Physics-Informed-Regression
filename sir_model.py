
def SIR(x, t):
    
    global beta
    global gamma
    
    S, I, R = x
    
    N = S + I + R
    
    dSdt = -beta/N * I * S
    dIdt = beta/N * I * S - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dIdt, dRdt]

beta = 0.30
gamma = 1/9

import numpy as np
from scipy.integrate import odeint

x0 = [5500000, 100000, 0]
t = np.arange(0, 100)
X = odeint(SIR, x0, t)



# For graphing
import plotly.express as px # For plotting and visualization
import plotly.io as pio
pio.renderers.default = "browser" # This set the default render as browser (This is not necessary if not using Spyder) 


fig = px.line(X)
fig.show()




