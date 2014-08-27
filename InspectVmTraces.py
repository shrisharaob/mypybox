from Startup import *
plt.ion()
def pltvm(vm, nNeurons, n):
    for i in range(n):
        id = np.random.randint(0, nNeurons, 1)
        plt.plot(vm[:, id])
        plt.draw()
        plt.title('%s' %(id,))
        plt.waitforbuttonpress()
        plt.clf()


