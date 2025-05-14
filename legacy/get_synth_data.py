# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:08:13 2021

@author: alboa
"""

import sim_functions as sf
import matplotlib.pyplot as plt
from models import SIR
from models import S3I3R
import numpy as np

def Create_synth_data(X_0: np.array, 
                      mp: list,
                      t: list, 
                      model=SIR,
                      T: list = [], 
                      noise_var: float = 0.1 # 
                      ):
    '''

    Parameters
    ----------
    X_0 : np.array
        Initial values                      
    mp : list
        model parameters.
    model: function, optional.
        Default is SIR. can also be set to S3I3R
    t : list
        Sample points.
    T : list, optional 
        vaccination list if expanded model is chosen. default is no vaccinations
    noise_var : float, optional
        
    
        

    Returns
    -------
    t : 
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    from models import SIR
    from models import S3I3R
    
    
    if model == SIR:
        if len(mp)!=2:
            print("Incorrect model parameter dimension! (must be 1x3)")
        if len(X_0) !=3:
            print("Incorrect initial value dimension! (must be 1x3) ")
        SIR = sf.SimulateModel(t,X_0,mp, model=SIR)
       
       #add noise and transform to daily data points
        SIR_noise = np.zeros((3,len(t)))
           
        for i in range(0,len(t)):
            daily_noise = np.random.normal(0.0, noise_var)*SIR[i,1] #size of noise is proportional to I
            if daily_noise <= 0:
                SIR_noise[:,i]=([SIR[i][0],SIR[i][1]+daily_noise,SIR[i][2]-daily_noise])
            else:
                SIR_noise[:,i]=([SIR[i][0]-daily_noise,SIR[i][1]+daily_noise,SIR[i][2]])
               
               
        return SIR_noise.transpose()
           
    if model == S3I3R:
        # translates default T=[] to actual zero vector of correct length
        if len(mp)!=8:
            print("Incorrect model parameter dimension! (must be 1x7)")
        if len(X_0) !=7:
            print("Incorrect initial value dimension! (must be 1x7) ")
        
        S3I3R = sf.SimulateModel(t,X_0,mp,model)
        S3I3R.transpose()
        # add noise (only to I1 category) and transform to daily data points. 
        S3I3R_noise = np.zeros((7,len(t)))
        
        for i in range(0,len(t)):
            print(i)
            daily_noise = np.random.normal(0.0, noise_var)*S3I3R[i,1] #size of noise is proportional to I
            
            if daily_noise <= 0:
                S3I3R_noise[:,i]=([S3I3R[i][0],S3I3R[i][1]+daily_noise,S3I3R[i][2]-daily_noise,S3I3R[i][3],S3I3R[i][4],S3I3R[i][5],S3I3R[i][6]])
            else:
                S3I3R_noise[:,i]=([S3I3R[i][0]-daily_noise,S3I3R[i][1]+daily_noise,S3I3R[i][2],S3I3R[i][3],S3I3R[i][4],S3I3R[i][5],S3I3R[i][6]])
          
                
        return S3I3R_noise
       
# TEST BASIC #
t=np.linspace(0,99,100)

X_0 = [10000, 100, 1]  
mp = [0.2, 0.11]

X_sir = Create_synth_data(X_0,mp,t)
plt.plot(t,X_sir)
plt.legend(['susceptible','infected','recovered'])
plt.show()

# TEST EXPANDED #
X_0 = [6000000,2000,20,2,0,0,0]
mp = [0.24,1/9, 1/7, 1/21,0.3,0.3,0.6,0]

X_s3i3r = Create_synth_data(X_0,mp,t,model=S3I3R,noise_var = 0.2)

plt.plot(t,X_s3i3r[0,:],t,X_s3i3r[1,:],t,X_s3i3r[2,:],t,X_s3i3r[3,:],t,X_s3i3r[4,:],t,X_s3i3r[5,:],t,X_s3i3r[6,:])
plt.legend(["S : susceptible","I1 : regular infected","I2 : ICU infected","I3 : respirator infected","R1 : regular recovered","R2 : vaccinated","R3 : dead"])
plt.ylim(0,10**4)
plt.show()