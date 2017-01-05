import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances
from SampleOpt import AMAOpt
from astLib import astCalc

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1619','J2228']

# DATA
fp = np.load('/data/ljo31b/EELs/esi/kinematics/inference/results_0.30_source_indous_vdfit_jul2016_J2228.npy')
l,m,u = fp
d = np.mean((l,u),axis=0)
dvl,dvs,dsigmal,dsigmas = d.T
vl,vs,sigmal,sigmas = m.T
s1 = sigmas.copy()
dsigmas = s1*0.05

# MODELS
ints = []
for i in range(len(names)):
    ints.append(np.load('/data/ljo31b/EELs/phys_models/models/interpolators/gNFW_gamma_12_'+names[i]+'.npy')[()])

x, dx = s1, dsigmas

gamma = pymc.Uniform('gamma',0,2.5 )
#tau = pymc.Uniform('tau',0,1.)
pars = [gamma]#,tau]
cov = np.array([0.3])#,0.3])


@pymc.deterministic
def logP(value=0.,p=pars):
    lp=0.
    GAMMA = gamma.value#, tau.value
    s2 = s1*0.
    #g = np.random.normal(loc=GAMMA,scale=TAU,size=len(names))
    for i in range(10):
        s2[i] = ints[i].eval(np.array([GAMMA]))
    chi2 = -0.5*(s2-s1)**2. / dsigmas**2.
    lp = chi2.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

# optimise first!

SS = AMAOpt(pars,[likelihood],[logP],cov=cov)
SS.sample(4000)
lp,trace,det = SS.result()


print 'results from optimisation:'
for i in range(len(pars)):
    pars[i].value = trace[-1,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

S = myEmcee.Emcee(pars+[likelihood],cov=cov/3.,nthreads=8,nwalkers=28)
S.sample(5000)
outFile = '/data/ljo31b/EELs/FP/inference/FP_infer_gNFW_gamma_HYPER'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = np.median(trace[2000:,:,i])
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])

GAMMA = gamma.value
s2 = s1*0.
for i in range(len(names)):
    s2[i] = ints[i].eval(np.array([GAMMA]))

pl.figure()
pl.scatter(s1,s1-s2,color='SteelBlue',s=40)

pl.figure()
pl.hist(dic['gamma'][1000:].ravel(),30,histtype='stepfilled',color='LightGray')
pl.xlabel('gNFW inner slope')
pl.show()

#np.save('/data/ljo31b/EELs/FP/inference/models/gNFW_gamma_12_sigmas',s2)
