import numpy as np
import pylab as plt


def Gaussian1D(x, mean, varianceOfGaussian):
    z1 = (1 / np.sqrt(2 * np.pi * varianceOfGaussian))
    denom = (2 * varianceOfGaussian * varianceOfGaussian)
    x = x - mean
    return  z1 * np.exp(-1 * x * x / (denom)) 

def Gaussian2D(x, y, varianceOfGaussian):
    z1 = (1 / np.sqrt(2 * np.pi * varianceOfGaussian))
    denom = (2 * varianceOfGaussian * varianceOfGaussian)
    return  z1 * z1 * np.exp(-1 * x * x / (denom)) * z1 * z1 * np.exp(-1 * y * y / (denom))

def ShortestDistOnCirc(point0, point1, perimeter):
  dist = np.abs(point0 - point1);
#  dist = dist / perimeter;
  dist = np.fmod(dist, perimeter)
  if(dist > 0.5):
    dist = 1.0 - dist;
  return dist;

def ConProb_new(xa, ya, xb, yb, patchSize, varianceOfGaussian):
  distX = ShortestDistOnCirc(xa, xb, patchSize);
  distY = ShortestDistOnCirc(ya, yb, patchSize);
  return Gaussian2D(distX, distY, varianceOfGaussian);

def ConProb_REFLECTING(xa, ya, xb, yb, patchSize, varianceOfGaussian):
  distX = np.abs(xa - xb)
  distY = np.abs(ya - yb)
  return Gaussian2D(distX, distY, varianceOfGaussian);

conProbType = 0

N_NEURONS = 900


#n = np.arange(0, 1.0, 0.05)
n = np.linspace(0.0, 1.0, 900)
y = np.zeros((n.size, n.size))
# for jj, i in enumerate(n):
# #    yCor = np.floor(float(i)/ 300.0) * (1.0 / 299.0)
# #    xCor = np.fmod(float(i), 300.0) * (1.0 / 299.0)
#     #tmpX = ShortestDistOnCirc(0., xCor, 1.0)
#     #tmpY = ShortestDistOnCirc(0., yCor, 1.0)
#     tmpX = np.abs(j-0.9)
#     tmpY = np.abs(i-0.8)
#     y[jj, kk] = Gaussian2D(tmpX, tmpY, .2)
    


for jj, i in enumerate(n):
    yCorA = np.floor(float(i)/ 300.0) * (1.0 / 299.0)
    xCorA = np.fmod(float(i), 300.0) * (1.0 / 299.0)
    for kk, j in enumerate(n):
        yCorB = np.floor(float(j)/ 300.0) * (1.0 / 299.0)
        xCorB = np.fmod(float(j), 300.0) * (1.0 / 299.0)
        if(conProbType == 0):
            tmpX = ShortestDistOnCirc(xCorA, xCorB, .1)
            tmpY = ShortestDistOnCirc(yCorA, yCorB, .1)
            y[jj, kk] = Gaussian2D(tmpX, tmpY, .2)
        else:
            distX = np.abs(j-0.9)
            distY = np.abs(i-0.8)
            y[jj, kk] = Gaussian2D(distX, distY, .1)







# for jj, i in enumerate(n):
#     for kk, j in enumerate(n):
#         if(conProbType == 0):
#             tmpX = ShortestDistOnCirc(0., j, 1.0)
#             tmpY = ShortestDistOnCirc(0., i, 1.0)
#             y[jj, kk] = Gaussian2D(tmpX, tmpY, .2)
#         else:
#             distX = np.abs(j-0.9)
#             distY = np.abs(i-0.8)
#             y[jj, kk] = Gaussian2D(distX, distY, .1)

plt.ion()
#plt.plot(n, y)
plt.imshow(y)
plt.ylim((0, n.size))
#plt.x
plt.colorbar()
plt.waitforbuttonpress()


    


