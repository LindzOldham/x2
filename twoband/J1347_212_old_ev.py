import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline
import SBBModels, SBBProfiles

pars = []
cov = []

# offsets
pars.append(pymc.Uniform('xoffset',-5.,5.,value=dic['xoffset'][a1,a2,a3]))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=dic['yoffset'][a1,a2,a3]))
cov += [0.1,0.1]

los = dict([('q',0.05),('pa',-180.),('re',0.1),('n',0.5)])
his = dict([('q',1.00),('pa',180.),('re',100.),('n',10.)])
covs = dict([('x',0.05),('y',0.05),('q',0.1),('pa',0.5),('re',0.5),('n',0.5)])
covlens = dict([('x',0.05),('y',0.05),('q',0.05),('pa',0.5),('b',0.2),('eta',0.1)])
lenslos, lenshis = dict([('q',0.05),('pa',-180.),('b',0.5),('eta',0.5)]), dict([('q',1.00),('pa',180.),('b',100.),('eta',1.5)])
gals = []
name = 'Galaxy 1'
p = {}
for key in 'x','y','q','pa','re','n':
    val = dic[name+' '+key][a1,a2,a3]
    if key == 'x' or key == 'y':
        lo,hi=val-10,val+10
    else:
        lo,hi= los[key],his[key]
    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
    p[key] = pars[-1]
    cov.append(covs[key])
gals.append(SBModels.Sersic(name,p))


# lensing is fixed except for the power law slope
lenses = []
name = 'Lens 1'
p = {}
for key in 'x','y','q','pa','b','eta':
    val = dic['Lens 1 '+key][a1,a2,a3]
    if key  in ['x','y']:
        lo,hi=val-5,val+5
    else:
        lo,hi= lenslos[key],lenshis[key]
    print lo,val,hi,key
    pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
    p[key] = pars[-1]
    cov.append(covlens[key])
lenses.append(MassModels.PowerLaw('Lens 1',p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=dic['extShear'][a1,a2,a3]))
cov.append(0.05)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180,value=dic['extShear PA'][a1,a2,a3]))
cov.append(1.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))

srcs = []
for name in ['Source 2','Source 1']:
    p = {}
    for key in 'q','re','n','pa':
        val = dic[name+' '+key][a1,a2,a3]
        lo,hi= los[key],his[key]
        pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
        p[key] = pars[-1]
        cov.append(covs[key])
    for key in 'x','y': 
        if name+' '+key in dic.keys():
            val = dic[name+' '+key][a1,a2,a3]
            lo,hi=val-10,val+10
            pars.append(pymc.Uniform('%s %s'%(name,key),lo ,hi,value=val ))
            p[key] = pars[-1] #+ lenses[0].pars[key]
            cov.append(covs[key])
        else:
            p[key] = srcs[0].pars[key]
    srcs.append(SBBModels.Sersic(name,p))


def likelihood():
    return 0.

def logprior():
    return 0.


def prior(pars):
    logp = 0.
    for par in pars:
        logp += par.logp
    return logp

import emcee
result = np.load('/data/ljo31/Lens/J1347/twoband_212_2_ctd')
lp,trace,dic,_ = result
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

for i in range(len(pars)):
    pars[i].value = trace[a1,a2,a3,i]

nwalkers, ndim, ntemps,nsteps = trace.shape[2], len(pars), trace.shape[1], trace.shape[0]
sampler = emcee.PTSampler(ntemps,nwalkers,ndim,likelihood,prior)

logL = lp.reshape((ntemps,nwalkers,nsteps)) - prior(pars)
ev = sampler.thermodynamic_integration_log_evidence(logL)
print ev
