import numpy as np
import pylab as plt

def QuadFit(x, y, alpha = 0.05):
#    X = np.array([np.ones((x.size)), x])
    X = np.array([np.ones((x.size)), x, x**2])
    X = np.transpose(X)
    coeff = np.dot(np.linalg.pinv(X), y)
    res = y - X.dot(coeff)
    varEstimate = np.sum(res ** 2) / (x.size - 3)
    covMatCoeff = np.linalg.inv(np.dot(np.transpose(X), X))
    c = 1.96 # implement student's t - distr lookup
    confInterval  = c * np.sqrt(covMatCoeff.diagonal())
    out = (np.flipud(coeff), confInterval, res)
    return out

if __name__ == '__main__':
    x = np.linspace(0, 1, 100000)
    a = 1
    b = 2
    c = 0.0
    noiseAmp = 0.0001
    y = c + b * x + a * x**2 + noiseAmp * np.random.randn(x.size)
    fit = QuadFit(x, y)
    print a, b, c
    print "fitted : ", fit[0]
    plt.plot(x, y, 'k.')
    plt.plot(x, np.polyval(fit[0], x), 'r')
    yu = np.polyval(fit[0] + fit[1], x)
    yd = np.polyval(fit[0] - fit[1], x)
    plt.plot(x, yu, 'r--')
    plt.plot(x, yd, 'r--')
#    plt.fill_between(x, yu, yd, 'g')
    # plt.figure()
    # plt.hist(fit[2], 25)

    plt.waitforbuttonpress()

