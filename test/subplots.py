import matplotlib.pyplot as plt
import numpy as np


def subplots(x, y, err=False, inred=None):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    ax1.set_xlabel('in')
    ax1.set_ylabel('out')
    ax1.plot(np.sort(x[:, 0]), np.sort(x[:, 0]), 'black')
    ax1.set_title('Micro Turbulance')

    ax2.set_xlabel('in')
    ax2.set_ylabel('out')
    ax2.plot(np.sort(x[:, 1]), np.sort(x[:, 1]), 'black')
    ax2.set_title('[Z]')

    ax3.set_xlabel('in')
    ax3.set_ylabel('out')
    ax3.plot(np.sort(x[:, 2]), np.sort(x[:, 2]), 'black')
    ax3.set_title('log g')

    ax4.set_xlabel('in')
    ax4.set_ylabel('out')
    ax4.plot(np.sort(x[:, 3]), np.sort(x[:, 3]), 'black')
    ax4.set_title('Teff')

    if inred is not None:
        ax1.scatter(x[:, 0][inred], y[:, 0][inred], color='red', s=100)
        ax2.scatter(x[:, 1][inred], y[:, 1][inred], color='red', s=100)
        ax3.scatter(x[:, 2][inred], y[:, 2][inred], color='red', s=100)
        ax4.scatter(x[:, 3][inred], y[:, 3][inred], color='red', s=100)

    if err is False:
        ax1.scatter(x[:, 0], y[:, 0])
        ax2.scatter(x[:, 1], y[:, 1])
        ax3.scatter(x[:, 2], y[:, 2])
        ax4.scatter(x[:, 3], y[:, 3])
    else:
        ax1.errorbar(x[:, 0], y[:, 0], yerr=y[:, 4],
                     fmt='|', ms=6, marker='o', lw=2, color='blue')
        ax2.errorbar(x[:, 1], y[:, 1], yerr=y[:, 5],
                     fmt='|', ms=6, marker='o', lw=2, color='blue')
        ax3.errorbar(x[:, 2], y[:, 2], yerr=y[:, 6],
                     fmt='|', ms=6, marker='o', lw=2, color='blue')
        ax4.errorbar(x[:, 3], y[:, 3], yerr=y[:, 7],
                     fmt='|', ms=6, marker='o', lw=2, color='blue')

# NGC6822:
in6822 = np.genfromtxt('../ngc6822/Catalogues/N6822-stellarpars-sam.txt')
tmp = in6822[:, 2: 10][:, 0::2]
inpar = np.array((tmp[:, 2], tmp[:, 3], tmp[:, 1], tmp[:, 0])).T

# inpar = np.genfromtxt('test/RSGinput.txt')
# inpar = np.genfromtxt('test/RSGinputall.txt')
outpar = np.genfromtxt('test/lineonly/lineonly-test3v18.txt')
# outpar = np.genfromtxt('test/lineonly/lineonly-test2v3.txt')


# subplots(inpar, outnoc, inred=0)
# for i, j in enumerate(inpar):
# for i in xrange(10):
i = 10
subplots(inpar, outpar, err=True, inred=i)
# subplots(inpar, outnoc, inred=4)
