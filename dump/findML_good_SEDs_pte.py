import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from stellarpop import tools

k_sun = 5.19
i_sun = 4.57
v_sun = 4.74

table = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckPhot_1src.fits')[1].data
lumb, lumv, Re, dlumb, dlumv, dRe, name = table['lum b'], table['lum v'], table['Re v'], table['lum b hi'], table['lum v hi'], table['Re v hi'], table['name']
lumk,dlumk = table_k['lum k'], table_k['lum k hi']
v,i,k = table['rest mag v'],table['rest mag i'],table_k['rest mag k']
dv,di,dk = table['rest mag v lo'],table['rest mag i lo'],table_k['rest mag k lo']
dvk = np.sqrt(dv**2.+dk**2.)
dvi = np.sqrt(dv**2.+di**2.)
vi0,vk0=v-i,v-k

#ii = np.where((vi<2.) & (vk<5.))

ages = ['0.010','0.125','0.250','0.375','0.500','0.625','0.750','0.875','1.000','1.250','1.500','1.700','1.750','2.000','2.200','2.250','2.500','2.750','3.000','3.250','3.500','3.750','4.000','4.250','4.500','4.750','5.000','5.250','5.500','5.750','6.000','7.000','8.000','9.000','10.00','12.00','15.00','20.00']
table_v = py.open('/data/ljo31/Lens/LensParams/Lumv_ages_1src.fits')[1].data
table_i = py.open('/data/ljo31/Lens/LensParams/Lumi_ages_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckLumk_ages_1src.fits')[1].data

grid_v = np.zeros((len(ages),12))
grid_i,grid_k = grid_v*0.,grid_v*0.
for a in range(len(ages)):
    grid_v[a] = table_v[ages[a]]
    grid_i[a] = table_i[ages[a]]
    grid_k[a] = table_k[ages[a]]
    
age_array = np.array([0.010,0.125,0.250,0.375,0.500,0.625,0.750,0.875,1.000,1.250,1.500,1.700,1.750,2.000,2.200,2.250,2.500,2.75,3.0,3.25,3.500,3.75,4.0,4.25,4.500,4.75,5.0,5.25,5.500,5.75,6.000,7.00,8.00,9.000,10.00,12.00,15.00,20.00])
bands = np.array(['F606W','F606W','F555W','F606W','F606W','F606W','F555W','F606W','F606W','F555W','F606W','F606W'])

interps_v,interps_i,interps_k = [],[],[]
for j in range(12):
    interps_v.append(splrep(age_array,grid_v[:,j]))
    interps_i.append(splrep(age_array,grid_i[:,j]))
    interps_k.append(splrep(age_array,grid_k[:,j]))


# these are the interpolator objects for the source/lens galaxies. Now, set up interpolator objects for the BC03 SEDs at different ages. We have the redshifts, so we can compute vi and vk colours at each age.
lums = []
kfilt = tools.filterfromfile('Kp_NIRC2')
ifilt = tools.filterfromfile('F814W_ACS')
v6filt = tools.filterfromfile('F606W_ACS')
v5filt = tools.filterfromfile('F555W_ACS')
grid_K,grid_I,grid_V6,grid_V5 = np.zeros(len(ages)), np.zeros(len(ages)),np.zeros(len(ages)),np.zeros(len(ages))
for a in range(len(ages)):
    age = ages[a]
    sed = tools.getSED('BC_Z=1.0_age='+age+'gyr')
    K_modrest = tools.ABFM(kfilt,sed,0.0) 
    I_modrest = tools.ABFM(ifilt,sed,0.0)
    V6_modrest = tools.ABFM(v6filt,sed,0.0)
    V5_modrest = tools.ABFM(v5filt,sed,0.0)
    grid_K[a],grid_I[a],grid_V6[a],grid_V5[a] = K_modrest, I_modrest, V6_modrest,V5_modrest
    
interps_K = splrep(age_array,grid_K)
interps_I = splrep(age_array,grid_I)
interps_V5 = splrep(age_array,grid_V5)
interps_V6 = splrep(age_array,grid_V6)

age_mls, ml_b, ml_v = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.4color',unpack=True,usecols=(0,4,5))
mlbmod, mlvmod = splrep(age_mls,ml_b), splrep(age_mls,ml_v)

pars = []
cov = []
pars.append(pymc.Uniform('log age 1',7,11,value=10.))
pars.append(pymc.Uniform('log age 2',7,11,value=10.))
pars.append(pymc.Uniform('log age 3',7,11,value=10.))
pars.append(pymc.Uniform('log age 4',7,11,value=10.))
pars.append(pymc.Uniform('log age 5',7,11,value=10.))
pars.append(pymc.Uniform('log age 6',7,11,value=10.))
pars.append(pymc.Uniform('log age 7',7,11,value=10.))
pars.append(pymc.Uniform('log age 8',7,11,value=10.))
pars.append(pymc.Uniform('log age 9',7,11,value=10.))
pars.append(pymc.Uniform('log age 10',7,11,value=10.))
pars.append(pymc.Uniform('log age 11',7,11,value=10.))
pars.append(pymc.Uniform('log age 12',7,11,value=10.))
cov += [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
optCov=np.array(cov)

@pymc.deterministic
def logP(value=0.,p=pars):
    lv=np.zeros(12)
    li,lk,model_i,model_v,model_k=lv*0.,lv*0.,lv*0.,lv*0.,lv*0.
    logT,divT = np.zeros(len(pars)),np.zeros(len(pars))
    for kk in range(len(pars)):
        logT[kk] = pars[kk].value
        divT[kk] = 10**logT[kk] * 1e-9
    for kk in range(12):
        if bands[kk] == 'F606W':
            model_v[kk] = splev(divT[kk],interps_V6).item()
        else:
            model_v[kk] = splev(divT[kk],interps_V5).item()
        model_i, model_k = splev(divT[kk],interps_I).item(), splev(divT[kk],interps_K).item()
        lv[kk] = splev(divT[kk],interps_v[kk]).item()
        li[kk] = splev(divT[kk],interps_i[kk]).item()
        lk[kk] = splev(divT[kk],interps_k[kk]).item()
    model_vi,model_vk = model_v-model_i, model_v-model_k
    v,i,k = -2.5*np.array(lv)+v_sun, -2.5*np.array(li)+i_sun, -2.5*np.array(lk) + k_sun
    vi,vk=v-i,v-k
    print moel_vi,vi
    print model_vk,vk
    resid = -0.5*np.sum((model_vi-vi)**2./dvi**2.) -0.5*np.sum((model_vk-vk)**2./dvk**2.)
    lp = resid
    return lp


@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=40,ntemps=3)
S.sample(10000)
outFile = '/data/ljo31/Lens/Analysis/1src_VKIage_v2_pte'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,0,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

#ftrace=trace.reshape((trace.shape[0]*trace.shape[1],trace.shape[2]))
#for i in range(len(pars)):
#    pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
#    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

#pl.figure()
#pl.plot(lp[200:])
#for i in range(trace.shape[-1]):
#    pl.figure()
#    pl.plot(trace[:,:,i])


lv=np.zeros(12)
li,lk,model_i,model_v,model_k,mlv,mlb=lv*0.,lv*0.,lv*0.,lv*0.,lv*0.,lv*0.,lv*0.
logT,divT = np.zeros(len(pars)),np.zeros(len(pars))
for kk in range(len(pars)):
    logT[kk] = pars[kk].value
    divT[kk] = 10**logT[kk] * 1e-9
for kk in range(12):
    if bands[kk] == 'F606W':
        model_v[kk] = splev(divT[kk],interps_V6).item()
    else:
        model_v[kk] = splev(divT[kk],interps_V5).item()
    model_i, model_k = splev(divT[kk],interps_I).item(), splev(divT[kk],interps_K).item()
    lv[kk] = splev(divT[kk],interps_v[kk]).item()
    li[kk] = splev(divT[kk],interps_i[kk]).item()
    lk[kk] = splev(divT[kk],interps_k[kk]).item()
    mlv[kk],mlb[kk] = splev(logT[kk],mlvmod), splev(logT[kk],mlbmod)
model_vi,model_vk = model_v-model_i, model_v-model_k
v,i,k = -2.5*np.array(lv)+v_sun, -2.5*np.array(li)+i_sun, -2.5*np.array(lk) + k_sun
vi,vk=v-i,v-k


mass = lv+np.log10(mlv)

for m in range(len(mass)):
    print name[m],' & ', '%.2f'%pars[m].value, ' & ', '%.2f'%mass[m], ' & ', '%.2f'%mlv[m], r'\\'

# save masses, mlv, age
ages = np.zeros(mass.size)
for i in range(len(mass)):
    ages[i] = pars[i].value
data = np.column_stack((ages,mass,mlv,vi,vk))
np.save('/data/ljo31/Lens/LensParams/InferredAges_1src_pte',data)
'''
# also uncertainties!!!
burnin=100
f = trace[burnin:].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[1],trace[burnin:].shape[2]))
logTs,lvs,lis,lks,mlvs,mlbs,vs,iis,ks = f*0.,f*0.,f*0.,f*0.,f*0.,f*0.,f*0.,f*0.,f*0.
for j in range(0,len(f)):
    p = f[j]
    logT,divT = np.zeros(len(pars)),np.zeros(len(pars))
    lv=np.zeros(12)
    li,lk,mlv,mlb=lv*0.,lv*0.,lv*0.,lv*0.
    for kk in range(len(pars)):
        logT[kk] = p[kk]
        divT[kk] = 10**logT[kk] * 1e-9
    for kk in range(12):
        lv[kk] = splev(divT[kk],interps_v[kk]).item()
        li[kk] = splev(divT[kk],interps_i[kk]).item()
        lk[kk] = splev(divT[kk],interps_k[kk]).item()
        mlv[kk],mlb[kk] = splev(logT[kk],mlvmod), splev(logT[kk],mlbmod)
    v,i,k = -2.5*np.array(lv)+v_sun, -2.5*np.array(li)+i_sun, -2.5*np.array(lk) + k_sun
    vi,vk=v-i,v-k
    logTs[j],lvs[j],lis[j],lks[j],mlvs[j],mlbs[j],vs[j],iis[j],ks[j] = logT,lv,li,lk,mlv,mlb,v,i,k

data = []
dic = dict([('logTs',logTs),('lvs',lvs),('lis',lis),('lks',lks),('mlvs',mlvs),('mlbs',mlbs),('vs',vs),('is',iis),('ks',ks)])
for quant in dic.keys():
    print keys
    np.save('/data/ljo31/Lens/Analysis/'+str(quant),dic[quant])
    lo,med,hi = np.percentile(dic[quant],[16,50,84],axis=0)
    data.append([lo,med,hi])
np.save('/data/ljo31/Lens/Analysis/InferredAges_1src_all_v2',np.array(data))
'''
