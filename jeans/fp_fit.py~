import numpy as np, pylab as pl, pyfits as py

re_norm, vd_norm, mu_norm = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_normals_100.dat').T
re_nugg, vd_nugg, mu_nugg = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_nuggets_100.dat').T

# scatter by Gaussian errors with sigma = 1% of observable
re_norm += np.random.randn(re_norm.size)*0.01*re_norm
vd_norm += np.random.randn(re_norm.size)*0.01*vd_norm
mu_norm += np.random.randn(re_norm.size)*0.01*mu_norm

re_nugg += np.random.randn(re_norm.size)*0.01*re_nugg
vd_nugg += np.random.randn(re_norm.size)*0.01*vd_nugg
mu_nugg += np.random.randn(re_norm.size)*0.01*mu_nugg

pl.figure()
pl.scatter(1.2*np.log10(vd_norm)+0.3*mu_norm, np.log10(re_norm),color='SteelBlue',s=40,label='normals')
pl.scatter(1.2*np.log10(vd_nugg)+0.3*mu_nugg, np.log10(re_nugg),color='Crimson',s=40,label='nuggets')
pl.xlabel('$1.2 \log\sigma + 0.3 \mu$')
pl.ylabel('$\log r_e$')
pl.legend(loc='lower right')

pl.figure()
pl.scatter(0.7*np.log10(vd_norm)+0.3*mu_norm, np.log10(re_norm),color='SteelBlue',s=40,label='normals')
pl.scatter(0.7*np.log10(vd_nugg)+0.3*mu_nugg, np.log10(re_nugg),color='Crimson',s=40,label='nuggets')
pl.xlabel('$0.7 \log\sigma + 0.3 \mu$')
pl.ylabel('$\log r_e$')
pl.legend(loc='lower right')

# nuggets generally have much brighter SBs (as more compact), then small sizes and fewer small VDs.

