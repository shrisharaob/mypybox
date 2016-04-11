for i in np.arange(25):
    mv0 = mv[:, 0]
    np.random.shuffle(mv0)
    coeff = np.polyfit(mv0, mv[:, 1], 2)
    x = np.linspace(0, 7, 100)
    y = np.polyval(coeff, x)
    plt.plot(x, y)
