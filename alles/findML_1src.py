import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from stellarpop import tools,distances

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
Al_hst = np.load('/data/ljo31/Lens/LensParams/Alambda_hst.npy')[()]
Al_keck = np.load('/data/ljo31/Lens/LensParams/Alambda_keck.npy')[()]
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
table = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckPhot_1src.fits')[1].data
v,i,k, dv1,di1,dk1, dv2, di2, dk2 = table['mag v'], table['mag i'],table_k['mag k'], table['mag v lo'], table['mag i lo'],table_k['mag k lo'], table['mag v hi'], table['mag i hi'],table_k['mag k hi']
## J2228 had the wrong V band!! It's F555W
v[-1],i[-1] = 20.88,18.91
Re,dRe1,dRe2,name = table['Re v'],table['Re v hi'],table['Re v lo'], table['name']
dRe,dv,di,dk = dRe1*0.,dv1*0.,dv1*0.,dv1*0.
for ii in range(name.size):
    dRe[ii] = np.min((dRe1[ii],dRe2[ii]))
    dv[ii] = np.min((dv1[ii],dv2[ii]))
    di[ii] = np.min((di1[ii],di2[ii]))
    dk[ii] = np.min((dk1[ii],dk2[ii]))
    v[ii] -= Al_hst[name[ii]][0]
    i[ii] -= Al_hst[name[ii]][1]
    k[ii] -= Al_keck[name[ii]]

dvk = np.sqrt(dv**2.+dk**2.) # get dvi from models (but probably not dvk as they are di
dvi = np.sqrt(dv**2.+di**2.)
vi,vk=v-i,v-k


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
kfilt = tools.filterfromfile('Kp_NIRC2')
ifilt = tools.filterfromfile('F814W_ACS')
v6filt = tools.filterfromfile('F606W_ACS')
v5filt = tools.filterfromfile('F555W_ACS')

interps_K, interps_I, interps_V,corr = [],[],[],[]
for n in name:
    z = sz[n]
    dl = dist.Dl(z)*cm_per_Mpc
    conv = 1./(4.*np.pi*dl**2.)
    conv *= 3.826e33
    corr.append(conv)
    SED = solarSpectra*conv
    grid_K,grid_I,grid_V = np.zeros(age_array.size),np.zeros(age_array.size),np.zeros(age_array.size)
    for a in range(age_array.size):
        grid_K[a] = tools.ABFM(kfilt,[wave,SED[a]],z)
        grid_I[a] = tools.ABFM(ifilt,[wave,SED[a]],z)
        if bands[n] == 'F555W':
            grid_V[a] = tools.ABFM(v5filt,[wave,SED[a]],z)
        else:
            grid_V[a] = tools.ABFM(v6filt,[wave,SED[a]],z)
    interps_K.append(splrep(age_array,grid_K))
    interps_I.append(splrep(age_array,grid_I))
    interps_V.append(splrep(age_array,grid_V))
    

pars = []
cov = []
for ii in range(12):
    #ars.append(pymc.Uniform('log age '+str(ii),6,9.92,value=9.)) # age of universe = 8.277 Gyr at z=0.55
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
        model_v[kk] = splev(logT[kk],interps_V[kk]).item()
        model_i[kk] = splev(logT[kk],interps_I[kk]).item()
        model_k[kk] = splev(logT[kk],interps_K[kk]).item()
    model_vi,model_vk = model_v-model_i, model_v-model_k
    resid = -0.5*(model_vi-vi)**2./dvi**2. -0.5*(model_vk-vk)**2./dvk**2.
    lp = resid.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=24,nwalkers=40)
S.sample(10000)
outFile = '/data/ljo31/Lens/Analysis/1src_VKIage_wideprior_new'
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
print v-model_v,i-model_i,k-model_k
masses = -0.4*(v-model_v)


print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{cccccccc}\hline'
print r' & $\log(T/yr)$ & $\log(M/M_{\odot})$ & $v-i$ & $v-k$ & $v-i$ (model) & $v-k$ (model) \\\hline'
for m in range(12):
    print name[m],' & ', '%.2f'%pars[m].value,  ' & ','%.2f'%masses[m], ' & ', '%.2f'%vi[m],' & ', '%.2f'%vk[m], ' & ', '%.2f'%model_vi[m],' & ', '%.2f'%model_vk[m], r'\\'
print r'\end{tabular}'
print r'\end{table}'

# save masses, mlv, age
ages = np.zeros(masses.size)
for i in range(len(masses)):
    ages[i] = pars[i].value
data = np.column_stack((ages,masses,vi,vk,model_vi,model_vk))
np.save('/data/ljo31/Lens/LensParams/InferredAges_1src_new',data)

# also uncertainties!!!
burnin=1000
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
logTs = np.zeros((f.shape[0]/100.,f.shape[1]))
Masses,vis,vks = logTs*0.,logTs*0.,logTs*0.

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
    model_vi,model_vk = model_v-model_i, model_v-model_k
    Mv,Mi,Mk =  -0.4*(v-model_v),-0.4*(i-model_i),-0.4*(k-model_k)
    logTs[ll],Masses[ll],vis[ll],vks[ll] = logT,Mv, model_vi, model_vk
    ll +=1

data = []
dic = dict([('logTs',logTs),('masses',Masses),('vis',vis),('vks',vks)])
for quant in dic.keys():
    print quant
    np.save('/data/ljo31/Lens/Analysis/Lensgal_'+str(quant),dic[quant])
    lo,med,hi = np.percentile(dic[quant],[16,50,84],axis=0)
    data.append([lo,med,hi])
np.save('/data/ljo31/Lens/Analysis/LensgalInferredAges_1src_all_new',np.array(data))
