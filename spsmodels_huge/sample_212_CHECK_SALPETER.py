import numpy as np, pylab as pl, cPickle
from linslens.ClipResult import clipburnin
from astLib import astCalc

import sys
name = sys.argv[1]
print name

lp, trace,det,_ = np.load('/data/ljo31b/EELs/inference/new/huge/result_212_CHECK_SALPETER_'+name)
a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0
pl.figure()
pl.plot(lp[2000:,0])
pl.show()

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

# data
sdssdata = np.load('/data/ljo31/Lens/LensParams/SDSS_phot_dereddened_dict_new.npy')[()]
vkidata = np.load('/data/ljo31/Lens/LensParams/VIK_phot_212_dict_huge_new.npy')[()]

# VIK dust corrections
Ahst = np.load('/data/ljo31/Lens/LensParams/Alambda_hst.npy')[()]
Akeck = np.load('/data/ljo31/Lens/LensParams/Alambda_keck.npy')[()]
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
data['v src'] = {'mag':v_src,'sigma':dvi_src}
data['v lens'] = {'mag':v_lens,'sigma':dvi_lens}

L = np.array([det[key][a1,0,a3] for key in ['logtau_V lens', 'tau lens', 'age lens','logZ lens']])
S = np.array([det[key][a1,0,a3] for key in ['logtau_V source', 'tau source', 'age source','logZ source']])
M = [det[key][a1,0,a3] for key in ['massL', 'massS']]
doexp = [True,False,False,True]
doexp = np.array(doexp)==True
L[doexp] = 10**L[doexp]
S[doexp] = 10**S[doexp]
l = np.atleast_2d([L])#.T
s = np.atleast_2d([S])#.T
ml,ms = M
mod = []
for f in sdssfilts:
    magl, mags = modelL.models[f].eval(l) -2.5*ml, modelS.models[f].eval(s) - 2.5*ms - 2.5*np.log10(mu)
    flux = 10**(-0.4*magl) + 10**(-0.4*mags)
    mod.append(-2.5*np.log10(flux))
vimodl = modelL.models[Vband].eval(l) -  modelL.models[Iband].eval(l)
vkmodl = modelL.models[Vband].eval(l) -  modelL.models[Kband].eval(l)
vimods = modelS.models[Vband].eval(s) -  modelS.models[Iband].eval(s)
vkmods = modelS.models[Vband].eval(s) -  modelS.models[Kband].eval(s)
vmods = modelS.models[Vband].eval(s)- 2.5*ms
vmodl = modelL.models[Vband].eval(l)- 2.5*ml
mod += [vimodl,vkmodl,vimods,vkmods,vmods,vmodl]

filtlist = sdssfilts+ ['v-i lens', 'v-k lens','v-i source', 'v-k source','v src', 'v lens']
# ein Tabular machen
for i in range(len(mod)):
    f = filtlist[i]
    print f, '& $','%.2f'%data[f]['mag'], r'\pm', '%.2f'%data[f]['sigma'], '$ & $', '%.2f'%mod[i], r'\\'

print ms, np.median(det['massS'][2000:].ravel())-np.percentile(det['massS'][2000:].ravel(),16), np.percentile(det['massS'][2000:].ravel(),84)-np.median(det['massS'][2000:].ravel())
