import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint

ii=1 # select which eel to plot

dat = np.loadtxt('/data/ljo31b/EELs/sizemass/re_allflux.dat')
r = dat[:,0]
f = dat[:,1:]
scale = 6.89268761
r *= 0.05*scale
logRe,logM,dlogRe,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_2src_huge_ljo.npy').T
logR = np.log10(r)

ff=f[:,ii]
nmod = splrep(logR,ff)
nnorm = splint(logR[0],logR[-1],nmod)

pl.figure()
pl.plot(np.log10(r), ff/nnorm,color='k',ls='--',label='lensing probability')
pl.xlabel('log (R$_e$/kpc)')
pl.ylabel('p(R$_e|$M$_{\star}$)')


result = np.load('/data/ljo31b/EELs/sizemass/sizemass_intrinsic_new_s2rc')
lp,trace,dic,_ = result
trace=trace[1000:]
ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
alpha,beta,sigma = np.percentile(ftrace,50,axis=0)
logr = beta*(logM[ii]-11.) + alpha
sigma2 = sigma**2. + dlogRe[ii]**2.
arg = (logR -logr)**2./sigma2 
norm = (2.*np.pi*sigma2)**0.5
pdf = np.exp(-0.5*arg) / norm

pl.plot(logR,pdf,color='k',ls=':',label='size probability')
pl.plot(logR,pdf*ff/nnorm,color='k',ls='-',label='total probability')
pl.xlim([-0.5,1.5])
pl.axvline(logRe[ii],color='k',ls='-.',label='measured size')
pl.legend(loc='upper right',fontsize=18)
pl.figtext(0.17,0.82,'J0901',fontsize=30)
