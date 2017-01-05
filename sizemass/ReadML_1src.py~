import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from stellarpop import tools,distances 

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
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
    
pars = np.zeros(12)
result = np.load('/data/ljo31/Lens/Analysis/1src_VKIage_wideprior_new')
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
for a in range(len(pars)):
    pars[a] = np.percentile(ftrace[:,a],50,axis=0)


model_i=np.zeros(12)
model_v,model_k=model_i*0.,model_i*0.
logT,divT = np.zeros(len(pars)),np.zeros(len(pars))
for kk in range(len(pars)):
    logT[kk] = pars[kk]
    divT[kk] = 10**logT[kk] * 1e-9
for kk in range(12):
    model_v[kk] = splev(logT[kk],interps_V[kk]).item()
    model_i[kk] = splev(logT[kk],interps_I[kk]).item()
    model_k[kk] = splev(logT[kk],interps_K[kk]).item()
model_vi,model_vk = model_v-model_i, model_v-model_k
Mv,Mi,Mk =  -0.4*(v-model_v),-0.4*(i-model_i),-0.4*(k-model_k)

masses = Mv.copy()

print r'\begin{table}[H]'
print r'\centering'
print r'\begin{tabular}{cccccccc}\hline'
print r' & $\log(T/yr)$ & $\log(M/M_{\odot})$ & $\Upsilon_v$ & $v-i$ & $v-k$ & $v-i$ (model) & $v-k$ (model) \\\hline'
for m in range(12):
    print name[m],' & ', '%.2f'%pars[m],  ' & ','%.2f'%masses[m], ' & ', '%.2f'%vi[m],' & ', '%.2f'%vk[m], ' & ', '%.2f'%model_vi[m],' & ', '%.2f'%model_vk[m], r'\\'
print r'\end{tabular}'
print r'\caption{Source galaxies, 211}'
print r'\end{table}'

# save masses, mlv, age
ages = np.zeros(masses.size)
for i in range(len(masses)):
    ages[i] = pars[i]
data = np.column_stack((ages,masses,vi,vk,model_vi,model_vk))
np.save('/data/ljo31/Lens/LensParams/InferredAges_1src_new',data)

# also uncertainties!!!
burnin=1000
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
logTs = np.zeros((f.shape[0]/100.,f.shape[1]))
mlvs,mlbs,Masses,lvs,vis,vks = logTs*0.,logTs*0.,logTs*0.,logTs*0.,logTs*0.,logTs*0.

ll=0
for j in range(0,len(f),100):
    p = f[j]
    logT,divT = np.zeros(len(pars)),np.zeros(len(pars))
    mlv,mass,lv=np.zeros(12),np.zeros(12),np.zeros(12)
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
np.save('/data/ljo31/Lens/Analysis/InferredAges_1src_all_new',np.array(data))
