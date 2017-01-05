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
pl.scatter(logM,logRe,color='Crimson',s=40)
plot_ellipses(logM,logRe,dlogM,dlogRe,rho,'Crimson')
#pl.fill_between(xfit,yfit,lo,color='LightPink',alpha=0.5)
#pl.fill_between(xfit,yfit,up,color='LightPink',alpha=0.5)
#xfit = np.linspace(10,13,20)
#pl.plot(xfit,yfit,color='Crimson',label='intrinsic')
'''vdWfit1 = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
pl.plot(xfit,vdWfit1,'k--',label = 'z=0.75')
vdWfit2 = 0.60 - 0.75*(10.+np.log10(5.)) + 0.75*xfit
pl.plot(xfit,vdWfit2,'k-.',label = 'z=0.25')
shenfit = np.log10(2.88e-6) + 0.56*xfit
pl.plot(xfit,shenfit,'k:',label = 'z=0')'''
vdWfit1 = 0.42 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
dvdW1 = 0.52 - 0.71*(10.+np.log10(5.)) + 0.71*xfit
dvdW2 = 0.32 - 0.71*(10.+np.log10(5.)) + 0.71*xfit

pl.plot(xfit,vdWfit1,'k-',label='normal galaxies')#,label = 'z=0.75')
#pl.plot(xfit,dvdW1,'k:')#,label = 'z=0.75')
#pl.plot(xfit,dvdW2,'k:')#,label = 'z=0.75')

B13 = (xfit - 10.3)/1.5
pl.plot(xfit,B13,'k--',label='nugget galaxies')
#vD15 = xfit - 10.7
#pl.plot(xfit,vD15,'k--',label='nugget galaxies')



pl.xlim([10.5,12])
pl.ylim([-0.4,1.9])
pl.xlabel('log(M$_{\star}$/M$_{\odot}$)')
pl.ylabel(r'log(R$_e$/kpc)')

#pl.plot(xfit,-11.36+1.05*xfit,color='Crimson',label='observed relation')
#pl.plot(xfit, 0.56*xfit+np.log10(2.88e-6),color='k',label='Shen 2003')
pl.legend(loc='upper left')
#pl.savefig('/data/ljo31/public_html/Lens/phys_models/intrinsic_v_observed_2src.png')
pl.show()

'''pl.figure()
pl.plot(lp[100:])

for key in dic.keys():
    pl.figure()
    pl.title(key)
    pl.plot(dic[key])
    #pl.figure()
    #pl.hist(dic[key].ravel(),30)
    #pl.title(key)
pl.show()

pl.figure(figsize=(7,15))
pl.subplot(311)
pl.hist(dic['alpha'][3000:].ravel(),30)
pl.title('alpha')
pl.subplot(312)
pl.hist(dic['beta'][3000:].ravel(),30)
pl.title('beta')
pl.subplot(313)
pl.hist(dic['sigma'][3000:].ravel(),30)
pl.title('sigma')
#pl.savefig('/data/ljo31/public_html/Lens/phys_models/inference_2src.png')
pl.show()

for key in dic.keys():
    pl.figure()
    pl.title(key)
    pl.plot(dic[key])
pl.show()

for key in dic.keys():
    pl.figure()
    pl.title(key)
    pl.hist(dic[key][1000:].ravel(),30)
pl.show()
'''
