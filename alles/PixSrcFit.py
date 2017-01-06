from imageSim import SBModels
import indexTricks as iT, pylab as pl, numpy as np, pyfits
import pymc
from SampleOpt import AMAOpt
from scipy import optimize

osrc = np.load('/data/ljo31/Lens/J1605/osrc_F814W.npy')
p = {}
pars = []
pars.append(pymc.Uniform('%s %s'%('Source 1','x'),210,230,value=220.5))
p['x'] = pars[-1]
pars.append(pymc.Uniform('%s %s'%('Source 1','y'),190,210,value=200.5))
p['y'] = pars[-1]
pars.append(pymc.Uniform('%s %s'%('Source 1','pa'),150,180,value=170))
p['pa'] = pars[-1]
pars.append(pymc.Uniform('%s %s'%('Source 1','q'),0.5,1,value=0.8))
p['q'] = pars[-1]
pars.append(pymc.Uniform('%s %s'%('Source 1','re'),1,40,value=15))
p['re'] = pars[-1]
pars.append(pymc.Uniform('%s %s'%('Source 1','n'),1,6,value=4))
p['n'] = pars[-1]
src = SBModels.Sersic('Source 1',p)
cov = [5,5,5,0.2,5,1] #zB!

y,x = iT.coords(osrc.shape)
ii = np.where(np.isnan(osrc)==False)
yc,xc = y[ii],x[ii]

osrcflt = osrc[ii].flatten()
rhs = osrcflt

def func(x,mod,rhs):
    return mod*x - rhs

@pymc.deterministic
def logP(value=0.,p=pars):
    model = src.pixeval(xc,yc)
    #op = model.T
    #fit,chi = optimize.nnls(op,rhs)
    fit = optimize.leastsq(func,0.1,args=(model,rhs))[0].item()
    model = model * fit
    ## currently no uncertainties to use...return the chi-squared.
    return np.sum(-0.5*(model - rhs)**2.)

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

optCov = np.array(cov)
   
for i in range(1):
    S = AMAOpt(pars,[likelihood],[logP]) #,cov=optCov)
    S.set_minprop(len(pars)*2)
    S.sample(90*len(pars)**2)

logp,trace,det = S.result()

coeff = []
for i in range(len(pars)):
    coeff.append(trace[-1,i])

coeff = np.asarray(coeff)

pars = coeff
o = 'npars = ['
for i in range(pars.size):
    o += '%f,'%(pars)[i]
o = o[:-1]+"]"

print o
