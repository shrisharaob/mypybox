import numpy as np


tau = np.arange(3, 12, 0.5)
np.savetxt('taulist.txt', tau, '%.1f')
