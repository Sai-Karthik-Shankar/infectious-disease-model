
#%%
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons

COLOR ='black'

mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

#%%
# Initial time conditions
T = 2000             # Total time of simulation (days)
dt = 1            # Time Step

#Initial Population Condition
N =  4000000              # Total Population
S0 = 3999000              # initial Susceptible population
E0 = 0               # Initial Exposed
I0 = 1000              # initial Infected Population
C0 = 0              # initial Critical Population
R0 = 0              # initial Recovered Population (ideally 0)
D0 = 0              # initial Dead Population (ideally 0)
Lockdown = 20
#R_0_start = 2
#R_0_end = 1
Beds_init = 1000
Birth = 1000


t = np.linspace(0,T,int(T/dt))
y0 = [S0, E0, I0, R0, C0, D0]

# SIR Parameters
Fa = 0.2           #Fatality Rate
Re = 1/4          #Recovery Rate Rate
Tr = 0.5          #Transmission Rate  (Depends on R0)
De = 1/12          #Infectious Death Rate
In = 1/3        # Incubation time
Bi = 0          #Birth Rate 
Dn = 0          #Natural Death Rate
p_IC = 0.05         # Probability of going critical from infected
p_CD = 0.05         # Probability of death from critical stage
I2C = 1/20           # 1 / time taken to become critical
C2R = 1/7          # 1 / time taken to recover from critical
C2D = 1/12          # time taken to die from critical
µ= 1/200


k = 1
s = 0.01
#%%
# function that Models an infectious Disease
def model(y ,t, N, Tr, Fa, Re, De, In, Lockdown, I2C, C2R, C2D, p_IC, p_CD):
        
        S = y[0]
        E = y[1]
        I = y[2]
        R = y[3]
        C = y[4]
        D = y[5]
        
        dSdt = -Tr*I*S/N + µ*R + Birth
        dEdt = Tr*I*S/N - In*E
        dIdt = In*E - (1-p_IC)*Re*I - I2C*p_IC*I
        dRdt = (1-Fa)*Re*I + C2R*(1-p_CD)*min(Beds(t),C) - µ*R
        dCdt = I2C*p_IC*I - C2R*(1-p_CD)*min(Beds(t),C) - C2D*p_CD*min(Beds(t),C) - max(0, C-Beds(t))
        dDdt = C2D*p_CD*min(Beds(t),C) + max(0, C-Beds(t))
        
        return [dSdt, dEdt, dIdt, dRdt, dCdt, dDdt]

        
        
        

def Beds(t):
        return Beds_init + s*t*Beds_init


def plotSIR(t, S, E, I, R, C, D):
        # plot results
        fig, ax = plt.subplots()
        fig.set_facecolor('#DDDDDA')
        ax.set_facecolor('#DDDDDA')
        plt.subplots_adjust(left=0.1, bottom=0.5)
        # Choose a seaborn-like style if available; fall back safely if not.
        preferred_styles = ['seaborn-dark', 'seaborn-darkgrid', 'seaborn', 'dark_background', 'classic']
        for _style in preferred_styles:
            if _style in plt.style.available:
                plt.style.use(_style)
                break
        else:
            try:
                import seaborn as sns
                sns.set_style("darkgrid")
            except Exception:
                plt.style.use('default')

        plt.title("Infectious disease Model")
        plt.xlabel("time (days)")
        plt.ylabel("population")
        plt.grid(color='#C2C8C5')  # bluish dark grey, but slightly lighter than background    
        s, = plt.plot(t, S,color = '#5874DC', linewidth = 2)
        #plt.annotate(int(S[-1]), xy=(T, S[-1])      # Annotate the current number of Susceptibles
        #        )
        e, = plt.plot(t, E, color = '#FA9284', linewidth = 2)
        #plt.annotate(int(E[-1]), xy=(T, E[-1])      # Annotate the current number of Susceptibles
        #        )
        i, = plt.plot(t, I, color = '#E06C78', linewidth = 2)
        #plt.annotate(int(I[-1]), xy=(T, I[-1])      # Annotate the current number of Infected
        #        )
        r, = plt.plot(t, R, color = '#6AAB9C', linewidth = 2)
        #plt.annotate(int(R[-1]), xy=(T, R[-1])      # Annotate the current number of Recovered
        #        )
        c, = plt.plot(t, C, color = '#E06C78', linewidth = 2, ls ='--')
        #plt.annotate(int(R[-1]), xy=(T, R[-1])      # Annotate the current number of Recovered
        #        )
        d, = plt.plot(t, D, color = '#384E78', linewidth = 2)
        #plt.annotate(int(D[-1]), xy=(T, D[-1])      # Annotate the current number of Dead
        #        )

       

        plt.legend(["Susceptible", "Exposed", "Infected", "Recovered","Critical", "Dead"])



        # Bottom Left

        bl = plt.axes([0.1, 0.2, 0.25, 0.15], facecolor='#C2C8C5')
        bl.grid(color='grey')  # bluish dark grey, but slightly lighter than background 
        
        b1, = bl.plot(t, C, color = '#E06C78', ls ='--')  
        b4, = bl.plot(t, np.repeat(Beds_init, len(t)), color = '#E06C78', ls ='--')
        bl.legend(["Critical"])

        #Bottom Middle
        bm = plt.axes([0.4, 0.2, 0.2, 0.15], facecolor='#C2C8C5')
        bm.grid(color='grey')  # bluish dark grey, but slightly lighter than background   
        b2, = bm.plot(t, N-D, color = '#FA9284', ls ='--')
        bm.legend(["No. Alive"])

        #Bottom Right
        br = plt.axes([0.65, 0.2, 0.25, 0.15], facecolor='#C2C8C5')
        br.grid(color='grey')  # bluish dark grey, but slightly lighter than background  
        b3, = br.plot(t[:-1], np.diff(D), color = '#384E78',)
        br.legend(["Death Per Day"])


        sl1 = plt.axes([0.25, .11, 0.50, 0.02])
        immunity = Slider(sl1, 'Immunity (days)', 0, 365, valinit=1/µ)
        def update(val):

                global µ
                # amp is the current value of the slider
                µ = 1/immunity.val 
                # update curve
                y = odeint(model,y0,t,args=(N, Tr, Fa, Re, De, In, Lockdown, I2C, C2R, C2D, p_IC, p_CD))
                s.set_ydata(y[:,0])
                e.set_ydata(y[:,1])
                i.set_ydata(y[:,2])
                r.set_ydata(y[:,3])
                c.set_ydata(y[:,4])
                d.set_ydata(y[:,5])
                b1.set_ydata(y[:,4])
                b3.set_ydata(np.diff(y[:,5]))
                plt.show()
        immunity.on_changed(update)

        plt.show()


#%%
def main():

        # solve ODE
        y = odeint(model,y0,t,args=(N, Tr, Fa, Re, De, In, Lockdown, I2C, C2R, C2D, p_IC, p_CD))

        S = y[:,0]
        E = y[:,1]
        I = y[:,2]
        R = y[:,3]
        C = y[:,4]
        D = y[:,5]

        plotSIR(t, S, E, I, R, C, D)




  
if __name__ == '__main__':
        # call main program
        main()


# %%
