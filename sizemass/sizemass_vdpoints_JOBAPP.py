import numpy as np, pylab as pl, pyfits as py, cPickle
import pymc, myEmcee_blobs as myEmcee
from astLib import astCalc
from scipy.interpolate import splrep, splev
from tools.EllipsePlot import *

dat = np.loadtxt('/data/ljo31b/EELs/sizemass/re_allflux.dat')
r = dat[:,0]
f = dat[:,1:]

logRe,logM,dlogRe,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_2src_huge_ljo.npy').T
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
names = sz.keys()
names.sort()
names = np.delete(names,6)
scales = np.array([astCalc.da(sz[name])*1e3*np.pi/180./3600. for name in names])
Re=10**logRe
dRe = dlogRe*Re

fluxes = Re*0.
for i in range(f.shape[1]):
    model = splrep(r*0.05*scales[i], f[:,i]/np.max(f[:,i]))
    fluxes[i] = splev(Re[i],model)


pars, cov = [], []
alpha = pymc.Uniform('alpha',-3,3)#,-10 )
beta = pymc.Uniform('beta',0,4)#,1.0 )
sigma = pymc.Uniform('sigma',0,1.)#,value=3.)
mu = pymc.Uniform('mu',-2.,2.)
tau = pymc.Uniform('tau',0,4.)


pars = [alpha,beta,sigma,mu,tau]
cov += [0.5,0.5,0.01,0.3,0.3]
optCov = np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    logrfunc = beta.value*(logM-11.) + alpha.value
    sigma2 = sigma.value**2. + dlogRe**2.
    arg = (logRe -logrfunc)**2./sigma2 
    marg = (logM-11. - mu.value)**2./tau.value**2.
    norm = (2.*np.pi*sigma2)**0.5
    m_norm = (2.*np.pi*tau.value**2.)**0.5
    prob = np.log(fluxes) - np.log(norm) - 0.5*arg - 0.5*marg - np.log(m_norm)
    lp = prob.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

#S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=80)
#S.sample(5000)
outFile = '/data/ljo31b/EELs/sizemass/sizemass_intrinsic_new_s2rc_massdist'
#f = open(outFile,'wb')
#cPickle.dump(S.result(),f,2)
#f.close()
#result = S.result()
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
trace=trace[1000:]
ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
for i in range(len(pars)):
    pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

alpha,beta,sigma,mu,tau= pars[0].value, pars[1].value,pars[2].value, pars[3].value,pars[4].value
xfit = np.linspace(8,14,20)

burnin=1000
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
fits=np.zeros((len(f),xfit.size))
for j in range(0,len(f)):
    alpha,beta,sigma,mu,tau = f[j]
    fits[j] = beta*(xfit-11.)+alpha

los,meds,ups = np.percentile(f,[16,50,84],axis=0)
los,ups=meds-los,ups-meds

yfit=meds[1]*(xfit-11.)+meds[0]
lo,med,up = xfit*0.,xfit*0.,xfit*0.
for j in range(xfit.size):
    lo[j],med[j],up[j] = np.percentile(fits[:,j],[16,50,84],axis=0)


# plot size-mass relation
pl.figure()

### size-mass relations
vdWfit1 = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
dvdW1 = 0.52 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
dvdW2 = 0.32 - 0.71*(10.+np.log10(5.)) + 0.71*xfit

pl.plot(xfit,vdWfit1,'k-',label = '$z=0.75$ (van der Wel+14)')#, z=0.75')
pl.plot(xfit,dvdW1,'k:')#,label = 'z=0.75')
pl.plot(xfit,dvdW2,'k:')#,label = 'z=0.75')

B13 = (xfit - 10.3)/1.5
vD15 = xfit - 10.7
#pl.plot(xfit,B13,'k-.',label='Barro+13')


### eels data
pl.scatter(logM,logRe,color='SteelBlue',s=30,label='EELs sources')
plot_ellipses(logM,logRe,dlogM,dlogRe,rho,'SteelBlue')


pl.xlim([10.45,12])
pl.ylim([-0.6,1.8])
pl.xlabel('log(M$_{\star}$/M$_{\odot}$)')
pl.ylabel(r'log(R$_e$/kpc)')


## taylor 2010
masses,radii = np.loadtxt('/data/ljo31/Lens/pylathon/sizemass/taylor2010.cat',unpack=True,usecols=(5,9))
radii = np.log10(radii)

pl.scatter(masses,radii,marker='o',s=30,color='Purple',label='z$\sim$0 (Taylor+10)')


# damjanov 2009
masses = np.array([1.16,3.14,0.59,1.25,3.18,0.56,0.67,1.06,2.85,1.34,5.94,4.68,4.65,2.15,2.07,4.06,3.58,3.52,3.58])
dm = np.array([0.27,0.43,0.27,0.39,0.44,0.15,0.24,0.30,0.98,0.53,0.95,0.16,0.40,0.86,0.89,0.94,1.10,0.51,1.48])
dm = dm/masses/np.log(10.)
radii = np.array([0.4,2.1,0.8,2.0,4.2,1.9,1.8,4.0,3.1,0.7,5.5,3.0,3.4,1.9,1.9,5.2,8.5,2.6,6.9])
dr = np.array([0.3,0.3,0.2,0.7,0.4,0.3,0.4,0.2,0.4,0.2,0.1,0.2,0.2,0.1,0.7,0.2,0.4,0.2,0.5])
dr = dr/radii/np.log(10.)
masses = np.log10(masses*1e11)
radii = np.log10(radii)

#pl.plot(xfit,vD15,'k--',label = '$z\sim2$ (van Dokkum+15)')


pl.errorbar(masses,radii,xerr=dm,yerr=dr,fmt='o',color='Orange',markeredgecolor='none')
pl.scatter(masses,radii,marker='o',s=30,color='Orange',label='z$\sim$1.5 (Damjanov+09)')

# van dokkum 2008
masses = np.array([5.04,2.10,4.06,2.48,3.22,2.74,0.95,3.47,1.41])
radii = np.array([0.76,0.92,0.78,1.89,0.93,1.42,0.47,2.38,0.49])
dm = np.array([1.00,0.68,1.07,0.40,1.08,0.77,0.11,1.00,0.50])
dr = [0.06,0.18,0.17,0.15,0.04,0.35,0.03,0.11,0.02]
dr = dr/radii/np.log(10.)
dm = dm/masses/np.log(10.)
masses = np.log10(masses*1e11)
radii = np.log10(radii)

pl.errorbar(masses,radii,xerr=dm,yerr=dr,fmt='o',color='SeaGreen',markeredgecolor='none')
pl.scatter(masses,radii,marker='o',s=30,color='SeaGreen',label='z$\sim$2 (van Dokkum+08)')


#pl.plot(xfit,-11.36+1.05*xfit,color='Crimson',label='observed relation')
#pl.plot(xfit, 0.56*xfit+np.log10(2.88e-6),color='k',label='Shen 2003')


pl.legend(loc='upper left',ncol=1,fontsize=15,scatterpoints=1)
#pl.savefig('/data/ljo31/public_html/Lens/phys_models/intrinsic_v_observed_2src.png')
pl.show()

