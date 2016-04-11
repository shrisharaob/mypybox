import numpy as np
import pylab as plt

N = 1000
K = 200
alpha = 0.0

adjMat = np.zeros((N, N))

k = float(K)
n = float(N)
p = k/n

pBi = alpha * 2 * p + (1 - alpha) * p ** 2 
pUni = 2 * (1 - alpha) * p * (1 - p)

for kNeuron in np.arange(N):
    for mNeuron in np.arange(kNeuron):
       # print kNeuron, mNeuron
        if(pBi >= np.random.rand()):
            adjMat[kNeuron, mNeuron] = 1
            adjMat[mNeuron, kNeuron] = 1
        else:
            if(pUni >= np.random.rand()): 
                if(np.random.rand() < 0.5):
                    adjMat[kNeuron, mNeuron] = 1
                else :
                    adjMat[mNeuron, kNeuron] = 1

print adjMat

#         if(p >= np.random.rand()): # both dirs
#             adjMat[kNeuron, mNeuron] = 1
#             if(alpha >= np.random.rand()):
#                 adjMat[mNeuron, kNeuron] = alpha # Pr(mNeuron --> kNeuron = 1 | k --> m = 1) ?
#         else: # Pr(kNeuron --> mNeuron = 0 | k --> n = 0)
#             if(q >= np.random.rand()):
#                 adjMat[mNeuron, kNeuron] = q
        

# for mNeuron in np.arange(1, N, 1):                
#     for kNeuron in np.arange(0, N-1, 1):
#         tmp = adjMat[mNeuron, kNeuron]
#         if(tmp != 1 and tmp != 0):
#             if(tmp >= np.random.rand()):
#                 adjMat[mNeuron, kNeuron] = 1

np.save('am', adjMat)
