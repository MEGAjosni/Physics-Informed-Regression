# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:08:13 2021

@author: alboa
"""

#import sim_functions as sf
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
        t,SIR = sf.SimulateModel(t,X_0,mp, model=SIR)
       
       #add noise and transform to daily data points
        SIR_noise = np.zeros((3,len(t)))
        count=0    
        for i in range(0,len(SIR)):
           if i%10 == 0 and i!= 0:
               daily_noise = np.random.normal(0.0, noise_var)*SIR[i,1] #size of noise is proportional to I
               if daily_noise <= 0:
                   SIR_noise[:,count]=([SIR[i][0],SIR[i][1]+daily_noise,SIR[i][2]-daily_noise])
               else:
                   SIR_noise[:,count]=([SIR[i][0]-daily_noise,SIR[i][1]+daily_noise,SIR[i][2]])
               count+=1    
               
        return t,SIR_noise.transpose()
           
    if model == S3I3R:
        # translates default T=[] to actual zero vector of correct length
        if len(mp)!=7:
            print("Incorrect model parameter dimension! (must be 1x7)")
        if len(X_0) !=7:
            print("Incorrect initial value dimension! (must be 1x7) ")
        
        t, S3I3R = sf.SimulateModel(t,X_0,mp,model)
        S3I3R.transpose()
        # add noise (only to I1 category) and transform to daily data points. 
        S3I3R_noise = np.zeros((7,len(t)+1))
        count=1
        for i in range(0,len(S3I3R)):
            if i%10 == 0 and i!= 0:
                daily_noise = np.random.normal(0.0, noise_var)*S3I3R[i,1] #size of noise is proportional to I
                
                if daily_noise <= 0:
                    S3I3R_noise[:,count]=([S3I3R[i][0],S3I3R[i][1]+daily_noise,S3I3R[i][2]-daily_noise,S3I3R[i][3],S3I3R[i][4],S3I3R[i][5],S3I3R[i][6]])
                else:
                    S3I3R_noise[:,count]=([S3I3R[i][0]-daily_noise,S3I3R[i][1]+daily_noise,S3I3R[i][2],S3I3R[i][3],S3I3R[i][4],S3I3R[i][5],S3I3R[i][6]])
                count+=1    
                
        return t,S3I3R_noise
       
# TEST BASIC #
'''
X_0 = [10000, 1000, 1]  
# wtf beta "virker" kun hvis den er giga ?
mp = [100, 0.11]

t,X = Create_synth_data(X_0,mp)
t=t[::10]
plt.plot(t[1:],X)
'''

# TEST EXPANDED #
X_0 = [6000000,1000,20,2,0,0,0]
mp = [1,1/9, 1/7, 1/21,0.3,0.3,0.6]

t,X = Create_synth_data(X_0,mp,model_type='expanded',noise_var = 0.1)
t=t[::10]
plt.plot(t,X[0,:],t,X[1,:],t,X[2,:],t,X[3,:],t,X[4,:],t,X[5,:],t,X[6,:])
plt.legend(["S : susceptible","I1 : regular infected","I2 : ICU infected","I3 : respirator infected","R1 : regular recovered","R2 : vaccinated","R3 : dead"])