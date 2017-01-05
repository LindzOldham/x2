import pymc,cPickle
from stellarpop import distances, tools
from scipy.interpolate import splrep, splint, splev
import myEmcee_blobs as myEmcee, pylab as pl, numpy as np

## we can only fit SSPs as we don't have so much data
dist = distances.Distance()
dist.h = 0.7
dist.OMEGA_M = 0.3
dist.OMEGA_L = 0.7
cm_per_Mpc = 3.08568e24
chabrier = np.load('/home/mauger/python/stellarpop/chabrier.dat')
age_array = chabrier['age']
age_array = np.log10(age_array)
age_array[0]=5.
wave=chabrier['wave']
solarSpectra = chabrier[6]
## (1) : make grids
gfilt = tools.filterfromfile('g_DES')
rfilt = tools.filterfromfile('r_DES')
ifilt = tools.filterfromfile('i_DES')

z = 1.165
dl = dist.Dl(z)*cm_per_Mpc
conv = 1./(4.*np.pi*dl**2.)
conv *= 3.826e33
SED = solarSpectra*conv
grid_g,grid_r,grid_i = np.zeros(age_array.size),np.zeros(age_array.size),np.zeros(age_array.size)
for a in range(age_array.size):
    grid_g[a] = tools.ABFM(gfilt,[wave,SED[a]],z)
    grid_r[a] = tools.ABFM(rfilt,[wave,SED[a]],z)
    grid_i[a] = tools.ABFM(ifilt,[wave,SED[a]],z)
   
mod_g = splrep(age_array,grid_g)    
mod_r = splrep(age_array,grid_r)
mod_i = splrep(age_array,grid_i)

t_univ = dist.age(z)
if t_univ>13.5:
    t_univ = 13.5
t_univ = t_univ-dist.age(10.)

tstart = dist.age(z)-dist.age(z+0.1)
tend = dist.age(z)-dist.age(15.)
print tstart, tend

#data - we just have two colours or three magnitudes. ATM we are only concerned with the source.
meds, diffs = np.load('/data/ljo31b/lenses/analysis/meds_diffs.npy')
magg,magr,magi,mug,mur,mui,reg,rer,rei,gr,gi = meds
dmagg,dmagr,dmagi,dmug,dmur,dmui,dreg,drer,drei,dgr,dgi = diffs
# dust corrections
Ag, Ar, Ai = 0.079, 0.053, 0.039
magg-=Ag
magr-=Ar
magi-=Ai

age = pymc.Uniform('age',tstart,tend)
mass = pymc.Uniform('mass',9.,12.5,value=11.)
pars = [age, mass]
cov = np.array([0.1,0.1])


@pymc.deterministic
def logL(value=0.,p=pars):
    logp = 0
    T,M = np.log10(age.value), mass.value
    modelg = splev(T,mod_g)
    modelr = splev(T,mod_r)
    modeli = splev(T,mod_i)
    model_gi, model_gr = modelg-modeli, modelg-modelr
    modelr -= 2.5*M
    logp = -0.5*(gi-model_gi)**2. / dgi**2. - 0.5*(gr-model_gr)**2. / dgr**2. - 0.5*(magr-modelr)**2./dmagr**2.
    return logp


@pymc.observed
def loglikelihood(value=0.,lp=logL):
    return lp

import myEmcee_blobs as myEmcee, pylab as pl
S = myEmcee.Emcee(pars+[loglikelihood],cov=cov,nthreads=1,nwalkers=30)
S.sample(1000)

outFile = '/data/ljo31b/lenses/analysis/ssp_chain'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]

pl.figure()
pl.subplot(311)
pl.plot(lp)
pl.title('logp')
pl.subplot(312)
pl.plot(dic['age'])
pl.title('age')
pl.subplot(313)
pl.plot(dic['mass'])
pl.title('mass')

pl.figure()
pl.subplot(211)
pl.hist(dic['age'][100:].ravel(),30,histtype='stepfilled')
pl.title('age')
pl.subplot(212)
pl.hist(dic['mass'][100:].ravel(),30,histtype='stepfilled')
pl.title('mass')
