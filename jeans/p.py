import pylab as pl, numpy as np


re_norm, vd_norm, mu_norm = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_normals_100_2re_shen.dat').T
re_nugg, vd_nugg, mu_nugg = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_nuggets_100_2re_shen.dat').T


pl.figure()
pl.scatter(np.log10(re_norm),np.log10(vd_norm),color='SteelBlue',s=40,label='normals')
pl.scatter(np.log10(re_nugg),np.log10(vd_nugg),color='Crimson',s=40,label='nuggets')
pl.xlabel('$\log r_e$')
pl.ylabel('$\log \sigma$')
pl.legend(loc='lower right')
