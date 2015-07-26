import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

outpar = np.genfromtxt('test/fullreg-test4v4.1.txt')
mti, zi, gi, ti = np.genfromtxt('test/RSGinput.txt')[::-1][1]
# outpar = np.genfromtxt('test/fullreg-test4v3.txt')
# mti, zi, gi, ti = np.genfromtxt('test/RSGinput.txt')[::-1][5]

mt, z, g, t = outpar.T

# Contour plot:
x = np.linspace(3400, 4400, 11)
y = np.delete(np.linspace(-1, 1, 21), (1, 19))
X, Y = np.meshgrid(x, y)
pdf = mlab.bivariate_normal(X, Y, np.std(t), np.std(z), np.mean(t), np.mean(z))
# ax1.contour(X, Y, pdf)

# Initial parameters:
# plt.plot(np.sort(t), np.linspace(zi, zi, 100), 'b', lw=2.)
# plt.plot(np.linspace(ti, ti, 100), np.sort(z), 'b', lw=2.)
# # Mean values:
# plt.plot(np.sort(t), np.linspace(np.mean(z), np.mean(z), 100), 'r', lw=2.)
# plt.plot(np.linspace(np.mean(t), np.mean(t), 100), np.sort(z), 'r', lw=2.)
# # Results:
# plt.scatter(t, z)
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
# means:
zbar = np.mean(z)
tbar = np.mean(t)
mtbar = np.mean(mt)
gbar = np.mean(g)
# Ranges:
trange = np.linspace(3000, 4500)
mtrange = np.linspace(1., 5.)
gzrange = np.linspace(-1, 1.)
# Panel 1:
ax1.plot(trange, np.linspace(zi, zi), 'b', lw=2.)
ax1.plot(np.linspace(ti, ti), gzrange, 'b', lw=2.)
ax1.set_xlabel('Teff')
ax1.set_ylabel('[Z]')
# Mean values:
ax1.plot(trange, np.linspace(zbar, zbar), 'r', lw=2.)
ax1.plot(np.linspace(tbar, tbar), gzrange, 'r', lw=2.)
# Results:
ax1.scatter(t, z)
ax1.axis((trange.min(), trange.max(), gzrange.min(), gzrange.max()))

# Panel 2:
ax2.plot(mtrange, np.linspace(zi, zi), 'b', lw=2.)
ax2.plot(np.linspace(mti, mti), gzrange, 'b', lw=2.)
ax2.set_xlabel('MicroTurb')
ax2.set_ylabel('[Z]')
# Mean values:
ax2.plot(mtrange, np.linspace(zbar, zbar), 'r', lw=2.)
ax2.plot(np.linspace(mtbar, mtbar), gzrange, 'r', lw=2.)
# Results:
ax2.scatter(mt, z)
ax2.axis((mtrange.min(), mtrange.max(), gzrange.min(), gzrange.max()))

# Panel 3:
ax3.plot(gzrange, np.linspace(zi, zi), 'b', lw=2.)
ax3.plot(np.linspace(gi, gi), gzrange, 'b', lw=2.)
ax3.set_xlabel('log g')
ax3.set_ylabel('[Z]')
# Mean values:
ax3.plot(gzrange, np.linspace(zbar, zbar), 'r', lw=2.)
ax3.plot(np.linspace(gbar, gbar), gzrange, 'r', lw=2.)
# Results:
ax3.scatter(g, z)
ax3.axis((gzrange.min(), gzrange.max(), gzrange.min(), gzrange.max()))

# Panel 4:
ax4.plot(mtrange, np.linspace(ti, ti), 'b', lw=2.)
ax4.plot(np.linspace(mti, mti), trange, 'b', lw=2.)
ax4.set_xlabel('MicroTurb')
ax4.set_ylabel('Teff')
# Mean values:
ax4.plot(mtrange, np.linspace(tbar, tbar), 'r', lw=2.)
ax4.plot(np.linspace(mtbar, mtbar), trange, 'r', lw=2.)
# Results:
ax4.scatter(mt, t)
ax4.axis((mtrange.min(), mtrange.max(), trange.min(), trange.max()))

# Panel 5:
ax5.plot(gzrange, np.linspace(ti, ti), 'b', lw=2.)
ax5.plot(np.linspace(gi, gi), trange, 'b', lw=2.)
ax5.set_xlabel('log g')
ax5.set_ylabel('Teff')
# Mean values:
ax5.plot(gzrange, np.linspace(tbar, tbar), 'r', lw=2.)
ax5.plot(np.linspace(gbar, gbar), trange, 'r', lw=2.)
# Results:
ax5.scatter(g, t)
ax5.axis((gzrange.min(), gzrange.max(), trange.min(), trange.max()))

# Panel 6:
ax6.plot(gzrange, np.linspace(mti, mti), 'b', lw=2.)
ax6.plot(np.linspace(gi, gi), mtrange, 'b', lw=2.)
ax6.set_xlabel('log g')
ax6.set_ylabel('MicroTurb')
# Mean values:
ax6.plot(gzrange, np.linspace(mtbar, mtbar), 'r', lw=2.)
ax6.plot(np.linspace(gbar, gbar), mtrange, 'r', lw=2.)
# Results:
ax6.scatter(g, mt)
ax6.axis((gzrange.min(), gzrange.max(), mtrange.min(), mtrange.max()))

plt.show()
