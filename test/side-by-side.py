import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.figure(figsize=(12, 12))
gs1 = gridspec.GridSpec(1, 3)
gs1.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.

lines = np.genfromtxt('lib/lines.txt')[:, 1]
lines.sort()
n6822 = np.genfromtxt('n6822-outspec.txt')
wave = n6822[:, 0]
ospec = n6822[:, 1:].T

mod = np.genfromtxt('mod-outspec.txt')
mwave = mod[:, 0]
mspec = mod[:, 1:].T

ax0 = plt.subplot(gs1[0])
ax1 = plt.subplot(gs1[1])
ax2 = plt.subplot(gs1[2])
for k, spec in enumerate(ospec):
    ax0.plot(wave, spec + k/2., 'black')
    ax1.plot(wave, spec + k/2., 'black')
    ax2.plot(wave, spec + k/2., 'black')
    # ax0.plot(wave, mspec[k] + k/2., 'red')
    # ax1.plot(wave, mspec[k] + k/2., 'red')
    # ax2.plot(wave, mspec[k] + k/2., 'red')

ax0.plot(np.linspace(lines[0], lines[0]), np.linspace(0, 7), lw=2., color='b')
ax1.plot(np.linspace(lines[3], lines[3]), np.linspace(0, 7), lw=2., color='b')
ax2.plot(np.linspace(lines[7], lines[7]), np.linspace(0, 7), lw=2., color='b')
ax0.set_ylabel(r'Normalised Flux')
ax1.set_xlabel(r'Wavelength ($\mu$m)')
ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax0.axis((1.18722, 1.1896, 0.6, 6.15))
ax1.axis((1.196115, 1.198495, 0.6, 6.15))
ax2.axis((1.209163, 1.211543, 0.6, 6.15))
ax0.get_xaxis().get_major_formatter().set_useOffset(False)
ax1.get_xaxis().get_major_formatter().set_useOffset(False)
ax2.get_xaxis().get_major_formatter().set_useOffset(False)
ax0.tick_params(axis='both', which='major', labelsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)
plt.show()
