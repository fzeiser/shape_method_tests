import matplotlib.pyplot as plt
import numpy as np

nld_cont = np.loadtxt("NLDcont.dat")
plt.figure()
# plt.semilogy(nld_cont[:, 0], nld_cont[:, 1])
bins = np.linspace(0, 7, num=11)
plt.hist(nld_cont[:, 0], weights=nld_cont[:, 1],
         bins=bins)
plt.show()
