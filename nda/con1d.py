import numpy as np
import pylab as plt

def Gaussian1D(x, mean, varianceOfGaussian):
    z1 = (1 / np.sqrt(2 * np.pi * varianceOfGaussian))
    denom = (2 * varianceOfGaussian * varianceOfGaussian)
    x = x - mean
    return  z1 * np.exp(-1 * x * x / (denom)) 

x = np.linspace(0.0, 1.0, 100)
y = np.linspace(0.0, 1.0, 100)
x0 = 0.9
y0 = 0.9
prob = np.zeros((len(x), len(y)))
K = 10.
for i, xi in enumerate(x):
    for j, yi in enumerate(y):
        prob[i, j] = Gaussian1D(xi, x0, 0.2) * Gaussian1D(yi, y0, 0.2)

zb = K / np.sum(prob, 0)
print zb.shape
#prob = zb * prob
#import pdb; pdb.set_trace()
plt.ion()
plt.imshow(prob)
plt.colorbar()
plt.waitforbuttonpress()



#fc[n] = np.sum(prob > np.random.rand(len(prob)))    


# print "sum(p) = ", prob.sum()
# print "Zb = K / sum(prob) = ", K / prob.sum()
# #plt.plot(x, prob * zb)
# #plt.waitforbuttonpress()
# print "sum(p) after prefac", prob.sum()
# print " avg connectivity = ",  
