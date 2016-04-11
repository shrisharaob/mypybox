from numpy import *
from pylab import *
import numpy as np

## setup parameters and state variables
T       = 100000                  # total time to simulate (msec)
dt      = 0.125               # simulation time step (msec)
time    = arange(0, T+dt, dt) # time array
t_rest  = 0                   # initial refractory time

## LIF properties
Vm      = zeros(len(time))    # potential (V) trace over time 
Rm      = 1                   # resistance (kOhm)
Cm      = 1                  # capacitance (uF)
tau_m   = Rm*Cm               # time constant (msec)
tau_ref = dt                   # refractory period (msec)
Vth     = 1                   # spike threshold (V)
V_spike = 1.5                 # spike delta (V)

## Input stimulus
I       = np.random.normal(0.0, 1.0, size = (len(time), ))                 # input current (A)
#I = 1.5 * ones((len(time), ))
nSpikes = []
## iterate over each time step
for i, t in enumerate(time): 
  if t > t_rest:
    Vm[i] = Vm[i-1] + (-Vm[i-1] + I[i] * Rm) / tau_m * dt
    if Vm[i] >= Vth:
      Vm[i] = V_spike
      t_rest = t + tau_ref
      nSpikes.append(t)

print 'firing rate: ', float(len(nSpikes)) / (T*1e-3)

## plot membrane potential trace  
plot(time, Vm, 'k', linewidth = 0.5)
# title('Leaky Integrate-and-Fire Example')
ylabel('Membrane Potential (V)')
xlabel('Time (msec)')
ylim([0,2])
show()

