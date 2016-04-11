from numpy import *
from pylab import *

## setup parameters and state variables
T       = 100                  # total time to simulate (msec)
dt      = 0.125               # simulation time step (msec)
time    = arange(0, T+dt, dt) # time array
t_rest  = zeros((2, ))      # initial refractory time

## LIF properties
Vm      = zeros((2, len(time)))    # potential (V) trace over time
Rm      = 1                   # resistance (kOhm)
Cm      = 10                  # capacitance (uF)
tau_m   = Rm*Cm               # time constant (msec)
tau_ref = dt #4                   # refractory period (msec)
Vth     = 1                   # spike threshold (V)
V_spike = 0.5                 # spike delta (V)

## Input stimulus
I       = 1.5                 # input current (A)
cur = zeros((2, ))
cur[0] = I
expDecay = exp(-dt / tau_m)
cur2 = []
J = 1.5 
## iterate over each time step
for i, t in enumerate(time):
    for kNeuron in range(2):
        if kNeuron == 1:
            cur[1] = cur[1] * expDecay
            cur2.append(cur[1])
        if t > t_rest[kNeuron]:
            Vm[kNeuron, i] = Vm[kNeuron, i-1] + (-Vm[kNeuron, i-1] + cur[kNeuron] * Rm) / tau_m * dt
            if Vm[kNeuron, i] >= Vth:
                Vm[kNeuron, i] += V_spike
                t_rest[kNeuron] = t + tau_ref
                if kNeuron == 0:
                    cur[1] = cur[1] + J 

## plot membrane potential trace
plot(time, Vm[0, :], label = 'neuron 1')
plot(time, Vm[1, :], label = 'neuron 2')
figure()
plot(cur2)
#title('Leaky Integrate-and-Fire Example')
ylabel('Membrane Potential (V)')
xlabel('Time (msec)')
ylim([0,2])
show()
