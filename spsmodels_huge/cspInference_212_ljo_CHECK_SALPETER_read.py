import pymc,sys,cPickle
from stellarpop import distances
from math import log10
from numpy import loadtxt
import numpy,glob
import myEmcee_blobs as myEmcee, pylab as pl
import numpy as np

name = sys.argv[1]
print name

modelname = name+'_lens_salpBC03.model'
filename = '/data/ljo31b/EELs/spsmodels/wide/'+modelname
f = open(filename,'rb')
modelL = cPickle.load(f)
f.close()

f = open(filename.replace('lens','source'),'rb')
modelS = cPickle.load(f)
f.close()

zl = modelL.redshifts[0]
zs = modelS.redshifts[0]
dist = distances.Distance()
dist.OMEGA_M = 0.3
dist.OMEGA_L = 0.7
dist.h = 0.7

t_univ = dist.age(zl)
if t_univ>13.5:
    t_univ = 13.5
t_univ = t_univ-dist.age(10.)

tstartL = dist.age(zl)-dist.age(zl+0.1)
tendL = dist.age(zl)-dist.age(15.)

tstartS = dist.age(zs)-dist.age(zs+0.1)
tendS = dist.age(zs)-dist.age(15.)

#data
sdssdata = numpy.load('/data/ljo31/Lens/LensParams/SDSS_phot_dereddened_dict_new.npy')[()]
vkidata = numpy.load('/data/ljo31/Lens/LensParams/VIK_phot_212_dict_huge_new.npy')[()] # make this

# VIK dust corrections
Ahst = numpy.load('/data/ljo31/Lens/LensParams/Alambda_hst.npy')[()]
Akeck = numpy.load('/data/ljo31/Lens/LensParams/Alambda_keck.npy')[()]
magnifications = np.load('/data/ljo31/Lens/LensParams/magnifications_212_huge.npy')[()]
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]

g,r,i,z,dg,dr,di,dz = sdssdata[name]
v_src,i_src,dv_src,di_src,vi_src,dvi_src, v_lens,i_lens,dv_lens,di_lens,vi_lens,dvi_lens, k_src, dk_src, k_lens, dk_lens = vkidata[name]
v_src, v_lens = v_src - Ahst[name][0], v_lens - Ahst[name][0]
i_src, i_lens = i_src - Ahst[name][1], i_lens - Ahst[name][1]
k_src,k_lens = k_src - Akeck[name], k_lens - Akeck[name]
mu = magnifications[name][0]
Vband = bands[name]+'_ACS'
sdssfilts = ['g_SDSS','r_SDSS','i_SDSS','z_SDSS']
Iband,Kband = 'F814W_ACS','Kp_NIRC2'

data = {}
data['g_SDSS'] = {'mag':g,'sigma':dg}
data['r_SDSS'] = {'mag':r,'sigma':dr}
data['i_SDSS'] = {'mag':i,'sigma':di}
data['z_SDSS'] = {'mag':z,'sigma':dz}
data['v-i lens'] = {'mag':v_lens-i_lens,'sigma':dvi_lens}
data['v-i source'] = {'mag':v_src-i_src,'sigma':dvi_src}
data['v-k lens'] = {'mag':v_lens-k_lens,'sigma':dk_lens}
data['v-k source'] = {'mag':v_src-k_src,'sigma':dk_src}
data['v lens'] = {'mag':v_lens,'sigma':dv_lens}
data['v source'] = {'mag':v_src,'sigma':dv_src}

ageL = pymc.Uniform('age lens',tstartL,tendL,value=3)
logZL = pymc.Uniform('logZ lens',-4.,log10(0.05))
tauL = pymc.Uniform('tau lens',0.04,5.1)
logVL = pymc.Uniform('logtau_V lens',-2.,log10(.1))

ageS = pymc.Uniform('age source',tstartS,tendS)
logZS = pymc.Uniform('logZ source',-4.,log10(0.05))
tauS = pymc.Uniform('tau source',0.04,5.1)
logVS = pymc.Uniform('logtau_V source',-2.,log10(.1))

massL = pymc.Uniform('massL',9.,12.5,value=11.)
massS = pymc.Uniform('massS',9.,12.5,value=11.5)

parsL = [logVL,tauL,ageL,logZL]
parsS = [logVS,tauS,ageS,logZS]
parsM = [massL,massS]
doexp = [True,False,False,True]
doexp = numpy.array(doexp)==True

pars = [logVL,tauL,ageL,logZL,logVS,tauS,ageS,logZS,massL,massS]
cov = np.array([0.1,0.03,0.5,0.03,0.1,0.03,0.5,0.03,0.3,0.3])

@pymc.deterministic
def logL(value=0.,L=parsL,S=parsS,M=parsM):
    logp = 0
    L,S = np.array(L), np.array(S)
    L[doexp] = 10**L[doexp]
    S[doexp] = 10**S[doexp]
    l = numpy.atleast_2d([L])#.T
    s = numpy.atleast_2d([S])#.T
    ml,ms = M
    for f in sdssfilts:
        magl, mags = modelL.models[f].eval(l) -2.5*ml, modelS.models[f].eval(s) - 2.5*ms - 2.5*np.log10(mu)
        flux = 10**(-0.4*magl) + 10**(-0.4*mags)
        flux = -2.5*np.log10(flux)
        logp += -0.5*(flux-data[f]['mag'])**2./data[f]['sigma']**2.
    vimodl = modelL.models[Vband].eval(l) -  modelL.models[Iband].eval(l)
    vkmodl = modelL.models[Vband].eval(l) -  modelL.models[Kband].eval(l)
    vimods = modelS.models[Vband].eval(s) -  modelS.models[Iband].eval(s)
    vkmods = modelS.models[Vband].eval(s) -  modelS.models[Kband].eval(s)
    vmodl = modelL.models[Vband].eval(l) - 2.5*ml
    vmods = modelS.models[Vband].eval(s) - 2.5*ms
    logp += -0.5*(vimodl - data['v-i lens']['mag'])**2. / data['v-i lens']['sigma']**2.
    logp += -0.5*(vkmodl - data['v-k lens']['mag'])**2. / data['v-k lens']['sigma']**2.
    logp += -0.5*(vimods - data['v-i source']['mag'])**2. / data['v-i source']['sigma']**2.
    logp += -0.5*(vkmods - data['v-k source']['mag'])**2. / data['v-k source']['sigma']**2.
    logp += -0.5*(vmodl - data['v lens']['mag'])**2. / data['v lens']['sigma']**2.
    logp += -0.5*(vmods - data['v source']['mag'])**2. / data['v source']['sigma']**2.
    return logp

@pymc.observed
def loglikelihood(value=0.,lp=logL):
    return lp

import myEmcee_blobs as myEmcee, pylab as pl
S = myEmcee.PTEmcee(pars+[loglikelihood],cov=cov,nthreads=24,nwalkers=30,ntemps=3)
print S.ndim
S.sample(10000)
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,0,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

S = myEmcee.PTEmcee(pars+[loglikelihood],cov=cov,nthreads=24,nwalkers=40,ntemps=3)#,filename='/data/ljo31b/EELs/inference/new/huge/prep_212_'+name+'.dat')    
S.sample(10000)
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,0,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

outFile = '/data/ljo31b/EELs/inference/new/huge/result_212_CHECK_SALPETER_'+str(name)
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()

pl.figure()
pl.plot(lp[200:,0])
#pl.show()
'''
# table
filtlist = sdssfilts+ ['v-i lens', 'v-k lens','v-i source', 'v-k source','v lens','v source']
# ein Tabular machen
for i in range(len(mod)):
    f = filtlist[i]
    print f, '& $','%.2f'%data[f]['mag'], r'\pm', '%.2f'%data[f]['sigma'], '$ & $', '%.2f'%mod[i], r'\\'
'''
