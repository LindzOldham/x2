import numpy as np, pylab as pl, pyfits as py, cPickle
import pymc, myEmcee_blobs as myEmcee
from astLib import astCalc
from scipy.interpolate import splrep, splev


r,f = np.loadtxt('/data/ljo31b/EELs/sizemass/re_flux.dat').T
modr = splrep(r*0.05*6.6,f)

logRe,logM,dlogRe,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_1src_huge_ljo.npy').T
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
names = sz.keys()
names.sort()
names = np.delete(names,6)
scales = np.array([astCalc.da(sz[name])*1e3*np.pi/180./3600. for name in names])
Re=10**logRe
dRe = dlogRe*Re
fluxes = splev(Re,modr)

pars, cov = [], []
alpha = pymc.Uniform('alpha',-20,0,-10 )
beta = pymc.Uniform('beta',0,20,1.0 )
sigma = pymc.Uniform('sigma',0,10.,value=5.)
pars = [alpha,beta,sigma]
cov += [0.5,0.5,0.01]
optCov = np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    logrfunc = beta.value*logM + alpha.value
    rfunc = 10**logrfunc # in kpc
    sigma2 = sigma.value**2. + dRe**2. 
    arg = (Re-rfunc)**2./sigma2
    norm = (2.*np.pi*sigma2)**0.5
    prob = np.log10(fluxes) - np.log10(norm) - 0.5*arg
    lp = prob.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=30)
S.sample(5000)
outFile = '/data/ljo31b/EELs/sizemass/sizemass_intrinsic'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
for i in range(len(pars)):
    pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)


pl.figure()
pl.plot(lp[100:])

'''for key in dic.keys():
    pl.figure()
    pl.title(key)
    pl.plot(dic[key])
    pl.figure()
    pl.hist(dic[key].ravel(),30)
    pl.title(key)
pl.show()'''

pl.figure(figsize=(7,15))
pl.subplot(311)
pl.hist(dic['alpha'].ravel(),30)
pl.title('alpha')
pl.subplot(312)
pl.hist(dic['beta'].ravel(),30)
pl.title('beta')
pl.subplot(313)
pl.hist(dic['sigma'].ravel(),30)
pl.title('sigma')
pl.savefig('/data/ljo31/public_html/Lens/phys_models/inference.png')
pl.show()

# plot size-mass relation
pl.figure()
pl.scatter(logM,logRe,color='Crimson',s=40)
xfit = np.linspace(10,13,20)
pl.plot(xfit,alpha.value+beta.value*xfit,color='SteelBlue',label='intrinsic relation')
pl.plot(xfit,-11.19+1.04*xfit,color='Crimson',label='observed relation')
pl.legend(loc='upper left')
pl.show()
