import numpy as np, pylab as pl, pyfits as py, cPickle
import pymc, myEmcee_blobs as myEMcee

r,f = np.loadtxt('/data/ljo31b/EELs/sizemass/re_flux.dat').T

# EELs data
logRe,logM,dlogRe,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_1src_huge_ljo.npy').T

x,y = logM, logRe 
sxx, syy = dlogM, dlogRe
sxx2, syy2 = sxx**2., syy**2.
sxy,syx = rho*sxx*syy, rho*syy*sxx

pars, cov = [], []
pars.append(pymc.Uniform('alpha',-20,10,-10 ))
pars.append(pymc.Uniform('beta',-10,20,1.0 ))
pars.append(pymc.Uniform('sigma',0,0.9,value=0.01))
pars.append(pymc.Uniform('tau',0.001,100,value=1))
pars.append(pymc.Uniform('mu',10,12,value=11))
cov += [0.5,0.5,0.01,1.,1.]
optCov = np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    ()


r,f = np.loadtxt('/data/ljo31b/EELs/sizemass/re_flux.dat').T
rr,ff = np.column_stack((r,r,r,r,r,r,r,r,r,r,r,r,r)), np.column_stack((f,f,f,f,f,f,f,f,f,f,f,f,f))
logRe,logM,dlogRe,dlogM,rho = np.load('/data/ljo31/Lens/LensParams/ReMass_1src_huge_ljo.npy').T
Re=10**logRe
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
names = sz.keys()
names.sort()
names = np.delete(names,6)
scales = np.array([astCalc.da(sz[name])*1e3*np.pi/180./3600. for name in names])
alpha, beta = 1.05,11.4

def func(alpha,beta,sigma):
    logrfunc = beta*logM + alpha
    rfunc = 10**logrfunc # in kpc
    rfunc /= (0.05*scales)
    prob = (rr-rfunc)**2./sigma**2.
    norm = (2.*np.pi*sigma**2.)**0.5
    prob = np.exp(-0.5*prob)/norm
    prob *= ff
    
