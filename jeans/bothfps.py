import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle

re_norm, vd_norm, mu_norm = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_normals_100_2re_Mstarlim.dat').T
re_nugg, vd_nugg, mu_nugg = np.loadtxt('/data/ljo31b/EELs/phys_models/FP_nuggets_100_2re_Mstarlim.dat').T

result = np.load('/data/ljo31b/EELs/inference/FP/jeansmodels_norm_2re_Mstarlim_1')
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
a,b,alpha = np.median(dic['a'][1000:].ravel()), np.median(dic['b'][1000:].ravel()), np.median(dic['alpha'][1000:].ravel())

ii=np.where(a*np.log10(vd_nugg)+b*mu_nugg>-1.8)

pl.figure()
pl.scatter(a*np.log10(vd_norm)+b*mu_norm, np.log10(re_norm),color='SteelBlue',s=40,label='normals')
pl.scatter(a*np.log10(vd_nugg[ii])+b*mu_nugg[ii], np.log10(re_nugg[ii]),color='Crimson',s=40,label='nuggets')
pl.xlabel('$'+'%.2f'%a+' \log\sigma + '+'%.2f'%b+' \mu$')
pl.ylabel('$\log r_e$')
pl.legend(loc='lower right')



# scatter by Gaussian errors with sigma = 1% of observable
f_norm = 10**(-0.4*mu_norm)
f_nugg = 10**(-0.4*mu_nugg)
df_norm = np.random.randn(re_norm.size)*0.05*f_norm
df_nugg = np.random.randn(re_nugg.size)*0.05*f_nugg
f_norm += np.random.randn(re_norm.size)*0.05*f_norm
f_nugg += np.random.randn(re_nugg.size)*0.05*f_nugg
mu_norm = -2.5*np.log10(f_norm)
mu_nugg = -2.5*np.log10(f_nugg)
dmu_norm = df_norm/f_norm
dmu_nugg = df_nugg/f_nugg

re_norm += np.random.randn(re_norm.size)*0.05*re_norm
vd_norm += np.random.randn(re_norm.size)*0.05*vd_norm

re_nugg += np.random.randn(re_norm.size)*0.05*re_nugg
vd_nugg += np.random.randn(re_norm.size)*0.05*vd_nugg

dre_norm = np.random.randn(re_norm.size)*0.05*re_norm
dvd_norm = np.random.randn(re_norm.size)*0.05*vd_norm

dre_nugg = np.random.randn(re_norm.size)*0.05*re_nugg
dvd_nugg = np.random.randn(re_norm.size)*0.05*vd_nugg


# nuggets generally have much brighter SBs (as more compact), then small sizes and fewer small VDs.

# now fit the FP for each population in turn. First, the normals
xx,yy,zz = np.log10(vd_nugg), mu_nugg.copy(), np.log10(re_nugg)
dxx,dyy,dzz = dvd_nugg/vd_nugg, dmu_nugg,dre_nugg/re_nugg
sxx,syy,szz = dxx**2.,dyy**2.,dzz**2.
syz,szy = 0.,0.
sxy,syx,sxz,szx = 0,0,0,0
syz,szy=0,0
# covariances
syz,szy = 0.85*dyy*dzz,0.85*dyy*dzz


