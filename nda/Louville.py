import numpy as np
import pylab as plt
import time
plt.ion()
plt.xlim((-3, +3))
plt.ylim((-3, +3))
t = np.arange(0, 10, 1)
x0 = -1 + 2 * np.random.rand(100)
p0 = -1 + 2 * np.random.rand(100)
xold = x0
pold = p0
figH, = plt.plot(x0, p0, 'r.')
plt.draw()
time.sleep(1)
for n, nt in enumerate(t):
    xnew = xold + np.cos(pold)
    pnew = pold - np.cos(xnew)
    pold = xnew
    xold = xnew
    figH.set_xdata(xnew)
    figH.set_ydata(pnew)
    plt.draw()
    time.sleep(1)
