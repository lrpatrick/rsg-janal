
# # Fine:
# # v2 = np.ma.masked_equal(chi, 0.0, copy=False)
# # idxf = np.unravel_index(np.argmin(v2), chi.shape)
# # print 'Fine Grid'
# # print 'Chisq min = ', v2.min()
# # print 'Corresponding Stellar parameters:'
# # print 'Teff, logg, MickyTurb, [Z]'
# # print teff[0][idxf[2]], logg[0][idxf[3]], mt[0][idxf[0]], abuns[0][idxf[1]]

# import matplotlib.pyplot as plt


# def cplot(x, y, z, n):
#     """
#         Wrap matplotlib's contout plot
#         np.meshgrid is a good way to do this!

#         Arguements:
#         x, y : x & y coordniates for figure
#         z : z-coordinate for contours
#     """
#     X, Y = np.meshgrid(x, y)
#     plt.figure()
#     cs = plt.contour(X, Y, z, n, colors='k')
#     plt.clabel(cs, inline=0, fontsize=10)
#     plt.title('logg vs [Z]')
#     plt.show()

# # z = vchi[mti, :, loggi, :]
# # Unused:


# def arridx(arr, value):
#     """
#         Return the index of a given value in an array
#         Arguments:
#         arr: numpy.ndarray
#             numpy N-Dimensional array
#         value : float
#             Value to search for

#         Output:
#         idx : tuple

#     """
#     v = np.ma.masked_where(arr != value, arr, copy=False)
#     idx = np.where(v.mask == False)
#     return idx

# delta = 0.025
# origin = 'lower'
# x = y = np.arange(-3.0, 3.01, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
# Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# Z = 10 * (Z1 - Z2)
# # Illustrate all 4 possible "extend" settings:

# fig, axs = plt.subplots(3, 2)
# for ax in axs.ravel():
#     X, Y = np.meshgrid()
#     cs = ax.contourf(X, Y, Z)


# def contour(x, y, z, n):
#     """
#         Wrap matplotlib's contout plot
#         np.meshgrid is a good way to do this!

#         Arguements:
#         x, y : x & y coordniates for figure
#         z : z-coordinate for contours
#     """
#     X, Y = np.meshgrid(x, y)
#     # plt.ion()
#     # plt.figure()
#     cs = plt.contour(X, Y, z, n, colors='k')
#     return cs
# plt.clabel(cs, inline=0, fontsize=10)


# coarse = chi[0::5, 0::2, 0::2, :]
# vc = np.ma.masked_equal(coarse, 0.0, copy=False)
# idx = np.unravel_index(np.argmin(vc), coarse.shape)
# xii, zi, gi, ti = idx
# xi = mpar.field('TURBS')[0]  # 21
# abuns = mpar.field('ABUNS')[0]  # 19
# g = mpar.field('GRAVS')[0]  # 9
# t = mpar.field('TEMPS')[0]  # 11


# # xi, z, g, t = mpar[0]
# n = (vc.min() + 1, vc.min() + 2, vc.min() + 3, vc.min() + 5, vc.min() + 10)
# f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
# ax1.contour(t, g, chi[xii, zi, :, :], n)
# ax2.contour(t, abuns, chi[xii, :, gi, :], n)
# ax3.contour(g, abuns, chi[xii, :, :, ti], n)
# ax4.contour(t, xi, chi[:, zi, gi, :], n)
# ax5.contour(g, xi, chi[:, zi, :, ti], n)
# ax6.contour(abuns, xi, chi[:, :, gi, ti], n)
# plt.show()
