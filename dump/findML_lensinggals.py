import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle

k_sun = 5.19
i_sun = 4.57
v_sun = 4.74

table = py.open('/data/ljo31/Lens/LensParams/LensinggalaxyPhot_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckGalPhot_1src_nomult.fits')[1].data
lumb, lumv, Re, dlumb, dlumv, dRe, name = table['lum b'], table['lum v'], table['Re v'], table['lum b hi'], table['lum v hi'], table['Re v hi'], table['name']
lumk,dlumk = table_k['lum k'], table_k['lum k hi']
v,i,k = table['rest mag v'],table['rest mag i'],table_k['rest mag k']
dv,di,dk = table['rest mag v lo'],table['rest mag i lo'],table_k['rest mag k lo']
dvk = np.sqrt(dv**2.+dk**2.)
dvi = np.sqrt(dv**2.+di**2.)
vi,vk=v-i,v-k

ii = np.where((name != 'J2228') & (name != 'J1323'))

ages = ['0.010','0.125','0.250','0.375','0.500','0.625','0.750','0.875','1.000','1.250','1.500','1.700','1.750','2.000','2.200','2.250','2.500','2.750','3.000','3.250','3.500','3.750','4.000','4.250','4.500','4.750','5.000','5.250','5.500','5.750','6.000','7.000','8.000','9.000','10.00','12.00','15.00','20.00']
table_v = py.open('/data/ljo31/Lens/LensParams/Lensinggalaxies_Lumv_ages_1src.fits')[1].data
table_i = py.open('/data/ljo31/Lens/LensParams/Lensinggalaxies_Lumi_ages_1src.fits')[1].data
table_k = py.open('/data/ljo31/Lens/LensParams/KeckGalLumk_ages_1src_nomult.fits')[1].data

grid_v = np.zeros((len(ages),12))
grid_i,grid_k = grid_v*0.,grid_v*0.
for a in range(len(ages)):
    grid_v[a] = table_v[ages[a]]
    grid_i[a] = table_i[ages[a]]
    grid_k[a] = table_k[ages[a]]
    
age_array = np.array([0.010,0.125,0.250,0.375,0.500,0.625,0.750,0.875,1.000,1.250,1.500,1.700,1.750,2.000,2.200,2.250,2.500,2.75,3.0,3.25,3.500,3.75,4.0,4.25,4.500,4.75,5.0,5.25,5.500,5.75,6.000,7.000,8.000,9.000,10.00,12.00,15.00,20.00])
interps_v,interps_i,interps_k = [],[],[]
for j in range(12):
    interps_v.append(splrep(age_array,grid_v[:,j]))
    interps_i.append(splrep(age_array,grid_i[:,j]))
    interps_k.append(splrep(age_array,grid_k[:,j]))

age_cols,vi_mod, vk_mod = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.2color',unpack=True,usecols=(0,5,7))
age_mls, ml_b, ml_v = np.loadtxt('/data/mauger/STELLARPOP/chabrier/bc2003_lr_m62_chab_ssp.4color',unpack=True,usecols=(0,4,5))
vimod, vkmod = splrep(age_cols,vi_mod), splrep(age_cols,vk_mod)
mlbmod, mlvmod = splrep(age_mls,ml_b), splrep(age_mls,ml_v)

pars = []
cov = []
pars.append(pymc.Uniform('log age',7,11,value=10.))
cov += [1.]
optCov=np.array(cov)

# modelling all source galaxies as having the same age
@pymc.deterministic
def logP(value=0.,p=pars):
    logT = pars[0].value
    divT = 10**logT *1e-9
    model_vi, model_vk = splev(logT,vimod), splev(logT,vkmod)
    lv = [splev(divT,mod).item() for mod in interps_v]
    li = [splev(divT,mod).item() for mod in interps_i]
    lk = [splev(divT,mod).item() for mod in interps_k]
    v,i,k = -2.5*np.array(lv)+v_sun, -2.5*np.array(li)+i_sun, -2.5*np.array(lk) + k_sun
    vi,vk=v-i,v-k
    resid = -0.5*np.sum((model_vi-vi[ii])**2./dvi[ii]**2.) -0.5*np.sum((model_vk-vk[ii])**2./dvk[ii]**2.)
    lp = resid
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.Emcee(pars+[likelihood],cov=optCov,nthreads=1,nwalkers=20)
S.sample(3000)
#outFile = '/data/ljo31/Lens/Analysis/sizemass_bestmodels_lensgals'
#f = open(outFile,'wb')
#cPickle.dump(S.result(),f,2)
#f.close()
result = S.result()
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a2,i]
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp)
pl.figure()
pl.plot(dic['log age'])
logT = dic['log age'][a1,a2]
model_vi,model_vk = splev(logT,vimod), splev(logT,vkmod)
divT = 10**logT *1e-9
lv = [splev(divT,mod).item() for mod in interps_v]
li = [splev(divT,mod).item() for mod in interps_i]
lk = [splev(divT,mod).item() for mod in interps_k]
v,i,k = -2.5*np.array(lv)+v_sun, -2.5*np.array(li)+i_sun, -2.5*np.array(lk) + k_sun
vi,vk=v-i,v-k


pl.figure()
pl.hist(vi[ii],10)
pl.axvline(model_vi,color='k',label='inferred colour')
pl.title('assuming all sources have same age: inferred log age '+'%.2f'%logT)
pl.xlabel('v-i')
#pl.legend('upper right')
pl.figure()
pl.hist(vk[ii],10)
pl.axvline(model_vk,color='k',label='inferred colour')
pl.title('assuming all sources have same age: inferred log age '+'%.2f'%logT)
pl.xlabel('v-k')
#pl.legend('upper right')

print 'mlv = ', splev(logT,mlvmod)
print 'mlb = ', splev(logT,mlbmod)

mlv,mlb = splev(logT,mlvmod), splev(logT,mlbmod)
mass = lv+np.log10(mlv)
print mass
pl.figure()
pl.hist(mass[ii],10)
