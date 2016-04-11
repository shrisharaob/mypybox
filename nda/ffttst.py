import numpy as np
import pylab as plt

dt = 0.001 # seconds
fs = 1 / dt
tStop = 10.0 # secs
N = int(tStop / dt)
n = np.arange(N)
f = 1.0 # signal freq
x = np.sin(2 * np.pi * 1 * n * dt)

nfft = N

X = np.fft.fft(x)

plt.figure()
plt.plot(n*dt, x)
plt.waitforbuttonpress()
plt.figure()
plt.plot(np.fft.fftfreq(N, dt), np.abs(X))

plt.waitforbuttonpress()
