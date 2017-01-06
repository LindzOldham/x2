import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from stellarpop import tools,distances

dist = distances.Distance()
#k_sun = 5.19
#i_sun = 4.57
#v_sun = 4.74
### solar spectrum
au = 1.496e11 # i don't know where this spectrum is! 
pc = 3.086e16
ratio = au/pc
Dm=5-5*np.log10(ratio)
spec = py.open('/data/ljo31b/MACSJ0717/data/sun_reference_stis_002.fits')[1].data
wl = spec['wavelength']
f = spec['flux']
spec = np.row_stack((wl,f))
###
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]

###
table = py.open('/data/ljo31/Lens/LensParams/Phot_2src.fits')[1].data
table2 = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckPhot_2src.fits')[1].data
table2_k = py.open('/data/ljo31/Lens/LensParams/KeckPhot_1src.fits')[1].data
lumv=np.zeros(12)

v,i,k,dv,di,dk,name = lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.
v[:9],i[:9],k[:9] = table['mag v'],table['mag i'],table_k['mag k']
dv[:9],di[:9],dk[:9] = np.min((table['mag v lo'],table['mag v hi']),axis=0),np.min((table['mag i lo'],table['mag i hi']),axis=0),np.min((table_k['mag k lo'],table_k['mag k hi']),axis=0)
dk[9],dk[10],dk[11] = np.min((table2_k['mag k lo'][0],table2_k['mag k hi'][0])), np.min((table2_k['mag k lo'][1],table2_k['mag k hi'][1])), np.min((table2_k['mag k lo'][5],table2_k['mag k hi'][5]))
di[9],di[10],di[11] = np.min((table2['mag i lo'][0],table2['mag i hi'][0])), np.min((table2['mag i lo'][1],table2['mag i hi'][0])),np.min((table2['mag i lo'][5],table2['mag i hi'][0]))
dv[9],dv[10],dv[11] = np.min((table2['mag v lo'][0],table2['mag v hi'][0])), np.min((table2['mag v lo'][1],table2['mag v hi'][1])),np.min((table2['mag v lo'][5],table2['mag v hi'][5]))
k[9],k[10],k[11] = table2_k['mag k'][0], table2_k['mag k'][1], table2_k['mag k'][5]
i[9],i[10],i[11] = table2['mag i'][0], table2['mag i'][1], table2['mag i'][5]
v[9],v[10],v[11] = table2['mag v'][0], table2['mag v'][1], table2['mag v'][5]
name = np.concatenate((table['name'],np.array(['J0837','J0901','J1218'])))
sort = np.argsort(name)
v,i,k,dv,di,dk,name = v[sort],i[sort],k[sort],dv[sort],di[sort],dk[sort],name[sort]
v[7],i[7] = 23.08,21.08 # j1347 improvement?
#v[8],i[8] = 22.27,20.20  # j1446 improvement?
#v[10],i[10] = 21.28,19.08 # j1606 without boxiness?
#k[3] = 19.97 # j1125 improvement???
#v[3],i[3] = 22.9,20.77 # j1125 improvement??
## bugbugbugbugbug!!!!
#i[2],i[3],i[4],i[6],i[7],i[8],i[9],i[10],i[11] = 19.72, 21.62, 20.20,19.98,21.63,20.90,20.57,19.68,19.63
## J2228 had the wrong V band!! It's F555W!!!
v[-1],i[-1] = 21.30,19.63


dvk = np.sqrt(dv**2.+dk**2.)
dvi = np.sqrt(dv**2.+di**2.)
vi,vk=v-i,v-k

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

pars = []
cov = []
for ii in range(12):
    #pars.append(pymc.Uniform('log age '+str(ii),6,9.92,value=9.)) # age of universe = 8.277 Gyr at z=0.55
    pars.append(pymc.Uniform('log age '+str(ii),6,11,value=9.)) # age of universe = 8.277 Gyr at z=0.55
    cov += [1.]
optCov=np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    model_i=np.zeros(12)
    model_v,model_k=model_i*0.,model_i*0.
    logT = np.zeros(len(pars))
    for kk in range(len(pars)):
        logT[kk] = pars[kk].value
    for kk in range(12):
        model_v[kk] = splev(pars[kk].value,interps_V[kk]).item()
        model_i[kk] = splev(pars[kk].value,interps_I[kk]).item()
        model_k[kk] = splev(pars[kk].value,interps_K[kk]).item()
    model_vi,model_vk = model_v-model_i, model_v-model_k
    resid = -0.5*(model_vi-vi)**2./dvi**2. -0.5*(model_vk-vk)**2./dvk**2.
    lp = resid.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=40)
S.sample(10000)
outFile = '/data/ljo31/Lens/Analysis/2src_VKIage_wideprior'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
#for i in range(len(pars)):
#    pars[i].value = trace[a1,a2,i]
#    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
for i in range(len(pars)):
    pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp[200:])
for i in range(trace.shape[-1]):
    pl.figure()
    pl.plot(trace[:,:,i])

model_i=np.zeros(12)
model_v,model_k=model_i*0.,model_i*0.
logT,divT = np.zeros(len(pars)),np.zeros(len(pars))
for kk in range(len(pars)):
    logT[kk] = pars[kk].value
    divT[kk] = 10**logT[kk] * 1e-9
for kk in range(12):
    model_v[kk] = splev(logT[kk],interps_V[kk]).item()
    model_i[kk] = splev(logT[kk],interps_I[kk]).item()
    model_k[kk] = splev(logT[kk],interps_K[kk]).item()
model_vi,model_vk = model_v-model_i, model_v-model_k
resid = -0.5*(model_vi-vi)**2./dvi**2. -0.5*(model_vk-vk)**2./dvk**2.
lp = resid.sum()
# need to calculate K corrections to get ML now...
masses,mlvs,lvs= v*0.,v*0.,v*0.
vinterps = []
for b in range(12):
    z = sz[name[b]]
    vcorr=age_array*0.
    for a in range(age_array.size):
        if bands[name[b]] == 'F555W':
            vcorr[a] = tools.ABFM(v5filt,[wave,solarSpectra[a]],0) - tools.ABFM(v5filt,[wave,solarSpectra[a]],z)
        else:
            vcorr[a] = tools.ABFM(v6filt,[wave,solarSpectra[a]],0) - tools.ABFM(v6filt,[wave,solarSpectra[a]],z)
    if bands[name[b]] == 'F555W':
        v_sun = tools.ABFM(v5filt,spec,z)+Dm
    else:
        v_sun = tools.ABFM(v6filt,spec,z)+Dm
    i_sun, k_sun = tools.ABFM(ifilt,spec,z)+Dm, tools.ABFM(kfilt,spec,z)+Dm
    interp_v = splrep(age_array,vcorr)
    vinterps.append(interp_v)
    Vcorr = splev(logT[b],interp_v)
    DL = dist.luminosity_distance(z)
    DM = 5.*np.log10(DL*1e6) - 5.
    lvs[b] = -0.4*(v[b] + Vcorr - DM - v_sun)
    mlvs[b] = splev(logT[b],mlvmod)
    masses[b] = lvs[b]+np.log10(mlvs[b])

print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{cccccccc}\hline'
print r' & $\log(T/yr)$ & $\log(M/M_{\odot})$ & $\Upsilon_v$ & $v-i$ & $v-k$ & $v-i$ (model) & $v-k$ (model) \\\hline'
for m in range(12):
    print name[m],' & ', '%.2f'%pars[m].value,  ' & ','%.2f'%masses[m], ' & ', '%.2f'%mlvs[m], ' & ', '%.2f'%vi[m],' & ', '%.2f'%vk[m], ' & ', '%.2f'%model_vi[m],' & ', '%.2f'%model_vk[m], r'\\'
print r'\end{tabular}'
print r'\end{table}'

# save masses, mlv, age
ages = np.zeros(masses.size)
for i in range(len(masses)):
    ages[i] = pars[i].value
data = np.column_stack((ages,masses,lvs,mlvs,vi,vk,model_vi,model_vk))
np.save('/data/ljo31/Lens/LensParams/InferredAges_2src_wideprior',data)

# also uncertainties!!!
burnin=5000
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
logTs = np.zeros((f.shape[0]/100.,f.shape[1]))
mlvs,mlbs,Masses,lvs,vis,vks = logTs*0.,logTs*0.,logTs*0.,logTs*0.,logTs*0.,logTs*0.

ll=0
for j in range(0,len(f),100):
    p = f[j]
    logT,divT = np.zeros(len(pars)),np.zeros(len(pars))
    mlv,mass,lv,model_v,model_i,model_k=np.zeros(12),np.zeros(12),np.zeros(12),np.zeros(12),np.zeros(12),np.zeros(12)
    for kk in range(len(pars)):
        logT[kk] = p[kk]
        divT[kk] = 10**logT[kk] * 1e-9
        model_v[kk] = splev(logT[kk],interps_V[kk]).item()
        model_i[kk] = splev(logT[kk],interps_I[kk]).item()
        model_k[kk] = splev(logT[kk],interps_K[kk]).item()
    for b in range(12):
        z=sz[name[b]]
        if bands[name[b]] == 'F555W':
            v_sun = tools.ABFM(v5filt,spec,z)+Dm
        else:
            v_sun = tools.ABFM(v6filt,spec,z)+Dm
        i_sun, k_sun = tools.ABFM(ifilt,spec,z)+Dm, tools.ABFM(kfilt,spec,z)+Dm
        Vcorr = splev(logT[b],vinterps[b])
        DL = dist.luminosity_distance(z)
        DM = 5.*np.log10(DL*1e6) - 5.
        lv[b] = -0.4*(v[b] + Vcorr - DM - v_sun)
        mlv[b] = splev(logT[b],mlvmod)
        mass[b] = lv[b]+np.log10(mlv[b])
    logTs[ll],mlvs[ll],Masses[ll],lvs[ll],vis[ll],vks[ll] = logT,mlv,mass,lv, model_v-model_i, model_v-model_k
    ll +=1

data = []
dic = dict([('logTs',logTs),('lvs',lvs),('mlvs',mlvs),('masses',Masses),('vis',vis),('vks',vks)])
for quant in dic.keys():
    print quant
    np.save('/data/ljo31/Lens/Analysis/'+str(quant),dic[quant])
    lo,med,hi = np.percentile(dic[quant],[16,50,84],axis=0)
    data.append([lo,med,hi])
np.save('/data/ljo31/Lens/Analysis/InferredAges_2src_all_wideprior',np.array(data))


pl.show()
