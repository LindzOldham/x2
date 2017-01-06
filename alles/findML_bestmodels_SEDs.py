import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from stellarpop import tools

k_sun = 5.19
i_sun = 4.57
v_sun = 4.74

#### sorting out two catalogues and putting best models together ###
table = py.open('/data/ljo31/Lens/LensParams/Phot_2src.fits')[1].data
table2 = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckPhot_2src.fits')[1].data
table2_k = py.open('/data/ljo31/Lens/LensParams/KeckPhot_1src.fits')[1].data

lumv=np.zeros(12)
Re,dlumv,dRe,name,lumk,lumi,dlumk,dlumi,di,dv,dk = lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.,lumv*0.
Re[:9] = table['Re v']
Re[9],Re[10],Re[11] = table2['Re v'][0], table2['Re v'][1], table2['Re v'][5]
dlumv[:9] = table['lum v hi']
dlumv[9],dlumv[10],dlumv[11] = table2['lum v hi'][0], table2['lum v hi'][1], table2['lum v hi'][5]
lumv[:9] = table['lum v']
lumv[9],lumv[10],lumv[11] = table2['lum v'][0], table2['lum v'][1], table2['lum v'][5]
dRe[:9] = table['Re v hi']
dRe[9],dRe[10],dRe[11] = table2['Re v hi'][0], table2['Re v hi'][1], table2['Re v hi'][5]
dlumi[:9] = table['lum i hi']
dlumi[9],dlumi[10],dlumi[11] = table2['lum i hi'][0], table2['lum i hi'][1], table2['lum i hi'][5]
lumi[:9] = table['lum i']
lumi[9],lumi[10],lumi[11] = table2['lum i'][0], table2['lum i'][1], table2['lum i'][5]
dlumk[:9] = table_k['lum k hi']
dlumk[9],dlumk[10],dlumk[11] = table2_k['lum k hi'][0], table2_k['lum k hi'][1], table2_k['lum k hi'][5]
lumk[:9] = table_k['lum k']
lumk[9],lumk[10],lumk[11] = table2_k['lum k'][0], table2_k['lum k'][1], table2_k['lum k'][5]
###colour uncertainties
dk[:9] = table_k['rest mag k lo']
dk[9],dk[10],dk[11] = table2_k['rest mag k lo'][0], table2_k['rest mag k lo'][1], table2_k['rest mag k lo'][5]
di[:9] = table['lum i']
di[9],di[10],di[11] = table2['rest mag i lo'][0], table2['rest mag i lo'][1], table2['rest mag i lo'][5]
di[:9] = table['rest mag i lo']
di[9],di[10],di[11] = table2['rest mag i lo'][0], table2['rest mag i lo'][1], table2['rest mag i lo'][5]

name = np.concatenate((table['name'],np.array(['J0837','J0901','J1218'])))
sort = np.argsort(name)
lumv,Re,dlumv,dRe,name = lumv[sort],Re[sort],dlumv[sort],dRe[sort],name[sort]
dv,di,dk,lumi,lumk,dlumi,dlumk = dv[sort],di[sort],dk[sort],lumi[sort],lumk[sort],dlumi[sort],dlumk[sort]
np.save('/data/ljo31/Lens/LensParams/bestmodels_quantities',np.column_stack((Re,dRe,lumv,dlumv,lumi,dlumi,lumk,dlumk,dv,di,dk)))
### done ###
dvi,dvk = np.sqrt(dv**2. + di**2.), np.sqrt(dv**2.+dk**2.)

### do the same for ages ####
ages = ['0.010','0.125','0.250','0.375','0.500','0.625','0.750','0.875','1.000','1.250','1.500','1.700','1.750','2.000','2.200','2.250','2.500','2.750','3.000','3.250','3.500','3.750','4.000','4.250','4.500','4.750','5.000','5.250','5.500','5.750','6.000','7.000','8.000','9.000','10.00','12.00','15.00','20.00']
table_v = py.open('/data/ljo31/Lens/LensParams/Lumv_ages_2src.fits')[1].data
table_i = py.open('/data/ljo31/Lens/LensParams/Lumi_ages_2src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckLumk_ages_2src.fits')[1].data
###########################################################################
table2_v = py.open('/data/ljo31/Lens/LensParams/Lumv_ages_1src.fits')[1].data
table2_i = py.open('/data/ljo31/Lens/LensParams/Lumi_ages_1src.fits')[1].data
table2_k = py.open('/data/ljo31/Lens/LensParams/KeckLumk_ages_1src.fits')[1].data

grid2_v = np.zeros((len(ages),12))
grid2_i,grid2_k = grid2_v*0.,grid2_v*0.
grid_v,grid_i,grid_k = grid2_v[:,:9]*0.,grid2_v[:,:9]*0.,grid2_v[:,:9]*0.

for a in range(len(ages)):
    grid_v[a] = table_v[ages[a]]
    grid_i[a] = table_i[ages[a]]
    grid_k[a] = table_k[ages[a]]
    grid2_v[a] = table2_v[ages[a]]
    grid2_i[a] = table2_i[ages[a]]
    grid2_k[a] = table2_k[ages[a]]
    
Grid_v, Grid_i, Grid_k = np.zeros(grid2_v.shape), np.zeros(grid2_v.shape),np.zeros(grid2_v.shape)
Grid_v[:,:9],Grid_i[:,:9], Grid_k[:,:9] = grid_v, grid_i, grid_k
Grid_v[:,9], Grid_v[:,10],Grid_v[:,11] = grid2_v[:,0], grid2_v[:,1],grid2_v[:,5]
Grid_i[:,9], Grid_i[:,10],Grid_i[:,11] = grid2_i[:,0], grid2_i[:,1],grid2_i[:,5]
Grid_k[:,9], Grid_k[:,10],Grid_k[:,11] = grid2_k[:,0], grid2_k[:,1],grid2_k[:,5]
Grid_v,Grid_k,Grid_i = Grid_v[:,sort],Grid_k[:,sort],Grid_i[:,sort]
### fertig! xxx

age_array = np.array([0.010,0.125,0.250,0.375,0.500,0.625,0.750,0.875,1.000,1.250,1.500,1.700,1.750,2.000,2.200,2.250,2.500,2.75,3.0,3.25,3.500,3.75,4.0,4.25,4.500,4.75,5.0,5.25,5.500,5.75,6.000,7.00,8.00,9.000,10.00,12.00,15.00,20.00])
bands = np.array(['F606W','F606W','F555W','F606W','F606W','F606W','F555W','F606W','F606W','F555W','F606W','F606W'])

interps_v,interps_i,interps_k = [],[],[]
for j in range(12):
    interps_v.append(splrep(age_array,Grid_v[:,j]))
    interps_i.append(splrep(age_array,Grid_i[:,j]))
    interps_k.append(splrep(age_array,Grid_k[:,j]))

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
    resid = -0.5*np.sum((model_vi-vi)**2./dvi**2.) -0.5*np.sum((model_vk-vk)**2./dvk**2.)
    lp = resid
    return lp


@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=40)
S.sample(10000)
outFile = '/data/ljo31/Lens/Analysis/2src_VKIage'
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
np.save('/data/ljo31/Lens/LensParams/InferredAges_2src',data)

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
    print quant
    np.save('/data/ljo31/Lens/Analysis/'+str(quant),dic[quant])
    lo,med,hi = np.percentile(dic[quant],[16,50,84],axis=0)
    data.append([lo,med,hi])
np.save('/data/ljo31/Lens/Analysis/InferredAges_2src_all',np.array(data))
