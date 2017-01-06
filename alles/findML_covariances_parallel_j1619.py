import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from stellarpop import tools,distances

### solar spectrum
au = 1.496e11
pc = 3.086e16
ratio = au/pc
Dm=5-5*np.log10(ratio)
spec = py.open('/data/ljo31b/MACSJ0717/data/sun_reference_stis_002.fits')[1].data
wl = spec['wavelength']
f = spec['flux']
spec = np.row_stack((wl,f))
###
dist = distances.Distance()
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']
chabrier = np.load('/home/mauger/python/stellarpop/chabrier.dat')
age_array = chabrier['age']
age_array = np.log10(age_array)
age_array[0]=5.
wave=chabrier['wave']
solarSpectra = chabrier[6]
## (1) : make grids
kfilt = tools.filterfromfile('Kp_NIRC2')
ifilt = tools.filterfromfile('F814W_ACS')
v6filt = tools.filterfromfile('F606W_ACS')
v5filt = tools.filterfromfile('F555W_ACS')

vinterps = []
i_sun,k_sun,v_sun = np.zeros(12),np.zeros(12),np.zeros(12)
for b in range(12):
    z = sz[name[b]]
    vcorr=age_array*0.
    for a in range(age_array.size):
        if bands[name[b]] == 'F555W':
            vcorr[a] = tools.ABFM(v5filt,[wave,solarSpectra[a]],0) - tools.ABFM(v5filt,[wave,solarSpectra[a]],z)
        else:
            vcorr[a] = tools.ABFM(v6filt,[wave,solarSpectra[a]],0) - tools.ABFM(v6filt,[wave,solarSpectra[a]],z)
    interp_v = splrep(age_array,vcorr)
    vinterps.append(interp_v)
    i_sun[b], k_sun[b] = tools.ABFM(ifilt,spec,z)+Dm, tools.ABFM(kfilt,spec,z)+Dm
    if bands[name[b]] == 'F555W':
        v_sun[b] = tools.ABFM(v5filt,spec,z)+Dm
    else:
        v_sun[b] = tools.ABFM(v6filt,spec,z)+Dm

interps_K, interps_I, interps_V = [],[],[]
for b in range(name.size):
    z = sz[name[b]]
    grid_K,grid_I,grid_V = np.zeros(age_array.size),np.zeros(age_array.size),np.zeros(age_array.size)
    for a in range(age_array.size):
        grid_K[a] = tools.ABFM(kfilt,[wave,solarSpectra[a]],z)
        grid_I[a] = tools.ABFM(ifilt,[wave,solarSpectra[a]],z)
        if bands[name[b]] == 'F555W':
            grid_V[a] = tools.ABFM(v5filt,[wave,solarSpectra[a]],z)
        else:
            grid_V[a] = tools.ABFM(v6filt,[wave,solarSpectra[a]],z)
    interps_K.append(splrep(age_array,grid_K))
    interps_I.append(splrep(age_array,grid_I))
    interps_V.append(splrep(age_array,grid_V))
    
age_mls,ml_b,ml_v = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.4color',unpack=True,usecols=(0,4,5))
mlbmod, mlvmod = splrep(age_mls,ml_b), splrep(age_mls,ml_v)

interps_K,interps_I,interps_V = interps_K[-2],interps_I[-2],interps_V[-2]

# load up Re and mag PDFs in 3 bands
def findML(vi,vk,dvi,dvk,v):
    # load up files
    pars = []
    cov = []
    pars.append(pymc.Uniform('log age ',6,11,value=9.)) # age of universe = 8.277 Gyr at z=0.55
    cov += [1.]
    optCov=np.array(cov)

    @pymc.deterministic
    def logP(value=0.,p=pars):
        logT = pars[0].value
        model_v = splev(logT,interps_V).item()
        model_i = splev(logT,interps_I).item()
        model_k = splev(logT,interps_K).item()
        model_vi,model_vk = model_v-model_i, model_v-model_k
        resid = -0.5*(model_vi-vi)**2./dvi**2. -0.5*(model_vk-vk)**2./dvk**2.
        lp = resid.sum()
        return lp

    @pymc.observed
    def likelihood(value=0.,lp=logP):
        return lp

    S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=2,nwalkers=40)
    S.sample(5000)
    result = S.result()
    lp,trace,dic,_ = result
    a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
    ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
    for i in range(len(pars)):
        pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
        #print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)
    logT = pars[0].value
    model_v = splev(logT,interps_V).item()
    model_i = splev(logT,interps_I).item()
    model_k = splev(logT,interps_K).item()
    model_vi,model_vk = model_v-model_i, model_v-model_k
    # need to calculate K corrections to get ML now...
    masses,mlvs,lvs= v*0.,v*0.,v*0.
    z = sz['J1619']
    Vcorr = splev(logT,vinterps[-2])
    DL = dist.luminosity_distance(z)
    DM = 5.*np.log10(DL*1e6) - 5.
    lvs = -0.4*(v + Vcorr - DM - v_sun[-2])
    mlvs = splev(logT,mlvmod)
    masses = lvs+np.log10(mlvs)
    return masses
   
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
table = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data
table_k = np.load('/data/ljo31/Lens/LensParams/KeckPhot_2src_new_dict.npy')[()]
dv,di,dk,name = table['mag v hi'], table['mag i hi'],table_k['mag k hi'],table['name']
dvk0,dvi0 = np.sqrt(dv**2.+dk**2.),np.sqrt(dv**2.+di**2.)
magk = table_k['mag k']
# order mag PDF data into useable form...
#vi,vk,dvi,dvk=np.zeros((12,330)),np.zeros((12,330)),np.zeros((12,330)),np.zeros((12,330))
vk,vi,dvk,dvi = np.zeros(200),np.zeros(200),np.zeros(200),np.zeros(200)
v=vk*0
name = 'J1619'
Res,mags = np.load('/data/ljo31/Lens/Analysis/ReMagPDFs_'+str(name)+'.npy')
rev,rei = Res.T
magv,magi =  mags.T
#rek,magk = np.load('/data/ljo31/Lens/Analysis/ReMagKeckPDFs_'+str(name[ii])+'.npy')
vi = magv[:200]-magi[:200]
vk = magv[:200]-magk[-2]
dvk = dvk0[-2]
dvi = dvi0[-2]
v = magv[:200]

import time
start = time.time()
masses = np.zeros(200)
for jj in range(200):
    print jj
    masses[jj] = findML(vi[jj],vk[jj],dvi,dvk,v[jj])
    print time.time()-start
    np.save('/data/ljo31/Lens/Analysis/ReMassTrace_200_asigo_j1619',masses)
