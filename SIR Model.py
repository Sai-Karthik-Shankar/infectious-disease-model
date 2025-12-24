
#%%
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
# function that contains the differential equations
def model(y, t, a, b, d, z, p):
    
    S=y[0]
    I=y[1]
    R=y[2]
    
    dSdt = p -b*S*I - d*S 
    dIdt = b*S*I + z*R -a*I*S
    dRdt = a*I*S + d*S - z*R
    
    return [dSdt, dIdt, dRdt]


#%%
def main():
    print("This is the main program")



    # initial condition
    S0 = 1000
    I0 = 1
    R0 = 0

    y0 = [S0,I0,R0]

    # parameters
    a = 0.005         #Rate of Cure/Death
    b = 0.005/S0   #Transmission Rate
    d = 0.0001         #Natural Death Rate
    z = 0.0001       #Reinfection Rate
    p = 0        #constant Birth Rate
    
    # time points
    t = np.linspace(0,10000,1000)

    # solve ODE
    y = odeint(model,y0,t,args=(a, b, d, z, p))

    S=y[:,0]
    I=y[:,1]
    R=y[:,2]

    # plot results

    plt.plot(t, S)
    plt.plot(t, I)
    plt.plot(t, R)

    plt.legend(["Susceptible", "Infected", "Recovered"])

    plt.show()
    # 3D-Plot
  
if __name__ == '__main__':
    # call main program
    main()


# %%
